using System.Net;
using System.Diagnostics;
using System.Text;
using System.Threading;
using MegaCrit.Sts2.Core.Debug;
using MegaCrit.Sts2.Core.Logging;
using STS2AIAgent.Game;

namespace STS2AIAgent.Server;

internal static class Router
{
    private const string ServiceName = "sts2-ai-agent";
    private const string ProtocolVersion = "2026-08-28-v2";
    private const string ModVersion = "0.8.0-rlsts2.46";
    private const string LogPrefix = "[STS2AIAgent.Router]";
    private const int MaxEventStreamTimeoutMs = 86_400_000;

    private static readonly HashSet<string> RevisionOptionalActions = new(
        StringComparer.OrdinalIgnoreCase)
    {
        "run_console_command",
        "save_and_quit",
        "abandon_run",
        "return_to_main_menu"
    };

    private static long _requestCounter;

    public static async Task HandleAsync(HttpListenerContext context, CancellationToken cancellationToken)
    {
        var seq = Interlocked.Increment(ref _requestCounter);
        var requestId = $"req_{DateTime.UtcNow:yyyyMMdd_HHmmss_ffff}_{seq}";
        var request = context.Request;
        var response = context.Response;
        var stopwatch = Stopwatch.StartNew();
        var statusCode = 500;

        try
        {
            Log.Info($"{LogPrefix} {requestId} {request.HttpMethod} {request.Url?.AbsolutePath}");

            if (request.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                request.Url?.AbsolutePath == "/health")
            {
                await WriteJsonAsync(response, 200, new
                {
                    ok = true,
                    request_id = requestId,
                    data = new
                    {
                        service = ServiceName,
                        mod_version = ModVersion,
                        protocol_version = ProtocolVersion,
                        game_version = ReleaseInfoManager.Instance.ReleaseInfo?.Version ?? "unknown",
                        status = "ready"
                    }
                });
                statusCode = 200;
                return;
            }

            if (request.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                request.Url?.AbsolutePath == "/state")
            {
                var generation = GameEventService.Instance.CaptureGeneration();
                var state = await GameThread.InvokeAsync(GameStateService.BuildStatePayload);
                GameEventService.Instance.ObserveState(state, generation);
                await WriteJsonAsync(response, 200, new
                {
                    ok = true,
                    request_id = requestId,
                    data = state
                });
                statusCode = 200;
                return;
            }

            if (request.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                request.Url?.AbsolutePath == "/actions/available")
            {
                var payload = await GameThread.InvokeAsync(GameStateService.BuildAvailableActionsPayload);
                await WriteJsonAsync(response, 200, new
                {
                    ok = true,
                    request_id = requestId,
                    data = payload
                });
                statusCode = 200;
                return;
            }

            if (request.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                request.Url?.AbsolutePath is string dataPath &&
                dataPath.StartsWith("/data/", StringComparison.OrdinalIgnoreCase))
            {
                var collectionPath = dataPath.Substring("/data/".Length);

                try
                {
                    var data = await GameThread.InvokeAsync(() => GameDataExportService.ExportCollection(collectionPath));
                    await WriteJsonAsync(response, 200, new
                    {
                        ok = true,
                        request_id = requestId,
                        data = data
                    });
                    statusCode = 200;
                    return;
                }
                catch (KeyNotFoundException)
                {
                    statusCode = 404;
                    await WriteErrorAsync(response, 404, "collection_not_found", $"Unknown data collection: {collectionPath}", requestId);
                    return;
                }
                catch (Exception ex)
                {
                    statusCode = 500;
                    await WriteErrorAsync(response, 500, "export_error", $"Failed to export {collectionPath}: {ex.Message}", requestId);
                    return;
                }
            }

            if (request.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                request.Url?.AbsolutePath == "/events/stream")
            {
                statusCode = await HandleEventStreamAsync(
                    request,
                    response,
                    cancellationToken);
                return;
            }

            if (request.HttpMethod.Equals("POST", StringComparison.OrdinalIgnoreCase) &&
                request.Url?.AbsolutePath == "/action")
            {
                var actionRequest = await JsonHelper.DeserializeAsync<ActionRequest>(request.InputStream, cancellationToken);
                if (actionRequest?.action == null)
                {
                    throw new ApiException(400, "invalid_request", "Request body must contain an action field.");
                }

                var actionResponse = await GameThread.InvokeAsync(async () =>
                {
                    using var suppression = NativeUiActionRecorder.Suppress();
                    var generation = GameEventService.Instance.CaptureGeneration();
                    var beforeState = GameStateService.BuildStatePayload();
                    GameEventService.Instance.ObserveState(beforeState, generation);
                    if (actionRequest.expected_state_revision == null &&
                        RequiresStateRevision(actionRequest.action))
                    {
                        throw new ApiException(
                            400,
                            "missing_state_revision",
                            "State-dependent actions require expected_state_revision.",
                            new
                            {
                                action = actionRequest.action,
                                current_state = beforeState
                            });
                    }
                    if (actionRequest.expected_state_revision is long expectedRevision &&
                        expectedRevision != beforeState.state_revision)
                    {
                        throw new ApiException(
                            409,
                            "stale_state",
                            "Observed state revision is no longer current.",
                            new
                            {
                                expected_state_revision = expectedRevision,
                                actual_state_revision = beforeState.state_revision,
                                current_state = beforeState
                            },
                            retryable: true);
                    }
                    var result = await GameActionService.ExecuteAsync(actionRequest);
                    GameEventService.Instance.PublishActionExecuted(
                        actionRequest, beforeState, result, generation);
                    return result;
                });
                await WriteJsonAsync(response, 200, new
                {
                    ok = true,
                    request_id = requestId,
                    data = actionResponse
                });
                statusCode = 200;
                return;
            }

            statusCode = 404;
            await WriteErrorAsync(response, statusCode, "not_found", "Route not found.", requestId);
        }
        catch (ApiException ex)
        {
            statusCode = ex.StatusCode;
            await WriteErrorAsync(response, ex.StatusCode, ex.Code, ex.Message, requestId, ex.Details, ex.Retryable);
        }
        catch (Exception ex)
        {
            Log.Error($"{LogPrefix} {requestId} Failed: {ex}");
            statusCode = 500;
            await WriteErrorAsync(response, statusCode, "internal_error", "Unhandled server error.", requestId);
        }
        finally
        {
            Log.Info($"{LogPrefix} {requestId} Completed {statusCode} in {stopwatch.ElapsedMilliseconds}ms");
            response.Close();
        }
    }

    public static Task WriteErrorAsync(
        HttpListenerResponse response,
        int statusCode,
        string code,
        string message,
        string? requestId = null,
        object? details = null,
        bool retryable = false)
    {
        return WriteJsonAsync(response, statusCode, new
        {
            ok = false,
            request_id = requestId ?? $"req_{DateTime.UtcNow:yyyyMMdd_HHmmss_ffff}_{Interlocked.Increment(ref _requestCounter)}",
            error = new
            {
                code,
                message,
                details,
                retryable
            }
        });
    }

    private static async Task WriteJsonAsync(HttpListenerResponse response, int statusCode, object payload)
    {
        var json = JsonHelper.Serialize(payload);
        var bytes = Encoding.UTF8.GetBytes(json);

        response.StatusCode = statusCode;
        response.ContentType = "application/json; charset=utf-8";
        response.ContentEncoding = Encoding.UTF8;
        response.ContentLength64 = bytes.LongLength;

        await response.OutputStream.WriteAsync(bytes);
    }

    private static async Task<int> HandleEventStreamAsync(
        HttpListenerRequest request,
        HttpListenerResponse response,
        CancellationToken cancellationToken)
    {
        var streamTimeout = ParseEventStreamTimeout(request);
        var generation = GameEventService.Instance.CaptureGeneration();
        var snapshot = await GameThread.InvokeAsync(GameStateService.BuildStatePayload);

        response.StatusCode = 200;
        response.ContentType = "text/event-stream";
        response.ContentEncoding = Encoding.UTF8;
        response.SendChunked = true;
        response.Headers["Cache-Control"] = "no-cache";
        response.Headers["Connection"] = "keep-alive";
        response.Headers["X-Accel-Buffering"] = "no";

        using var subscription = GameEventService.Instance.Subscribe(
            snapshot, generation);
        using var streamWaitCts = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken);
        var streamWaitToken = streamWaitCts.Token;

        try
        {
            await WriteSseCommentAsync(response, "stream opened");
            var heartbeat = Task.Delay(TimeSpan.FromSeconds(15), streamWaitToken);
            var streamDeadline = streamTimeout.HasValue
                ? Task.Delay(streamTimeout.Value, streamWaitToken)
                : Task.Delay(Timeout.InfiniteTimeSpan, streamWaitToken);

            while (!cancellationToken.IsCancellationRequested)
            {
                var waitForEvent = subscription.Reader.WaitToReadAsync(streamWaitToken).AsTask();
                var completedTask = await Task.WhenAny(
                    waitForEvent,
                    heartbeat,
                    streamDeadline);

                if (completedTask == streamDeadline)
                {
                    return 200;
                }

                if (completedTask == heartbeat)
                {
                    await WriteSseCommentAsync(response, "heartbeat");
                    heartbeat = Task.Delay(TimeSpan.FromSeconds(15), streamWaitToken);
                    continue;
                }

                if (!await waitForEvent)
                {
                    break;
                }

                while (subscription.Reader.TryRead(out var envelope))
                {
                    await WriteSseEventAsync(response, envelope);
                }
            }

            return 200;
        }
        catch (OperationCanceledException)
        {
            return 200;
        }
        catch (HttpListenerException)
        {
            // 客户端已断开连接。
            return 200;
        }
        catch (IOException)
        {
            // 客户端已断开连接。
            return 200;
        }
        catch (ObjectDisposedException)
        {
            // 响应流已经关闭。
            return 200;
        }
        finally
        {
            streamWaitCts.Cancel();
        }
    }

    private static bool RequiresStateRevision(string? action)
    {
        var normalized = action?.Trim();
        return !string.IsNullOrEmpty(normalized) &&
               !RevisionOptionalActions.Contains(normalized);
    }

    private static TimeSpan? ParseEventStreamTimeout(HttpListenerRequest request)
    {
        var raw = request.QueryString["timeout_ms"];
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        if (!int.TryParse(raw, out var timeoutMs) ||
            timeoutMs < 1 ||
            timeoutMs > MaxEventStreamTimeoutMs)
        {
            throw new ApiException(
                400,
                "invalid_request",
                $"timeout_ms must be between 1 and {MaxEventStreamTimeoutMs}.");
        }

        return TimeSpan.FromMilliseconds(timeoutMs);
    }

    private static async Task WriteSseEventAsync(HttpListenerResponse response, GameEventEnvelope envelope)
    {
        await WriteSseRawAsync(response, $"id: {envelope.event_id}\n");
        await WriteSseRawAsync(response, $"event: {envelope.type}\n");

        var json = JsonHelper.Serialize(envelope);
        foreach (var line in json.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n'))
        {
            await WriteSseRawAsync(response, $"data: {line}\n");
        }

        await WriteSseRawAsync(response, "\n");
        await response.OutputStream.FlushAsync();
    }

    private static async Task WriteSseCommentAsync(HttpListenerResponse response, string comment)
    {
        await WriteSseRawAsync(response, $": {comment}\n\n");
        await response.OutputStream.FlushAsync();
    }

    private static ValueTask WriteSseRawAsync(HttpListenerResponse response, string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        return response.OutputStream.WriteAsync(bytes);
    }
}
