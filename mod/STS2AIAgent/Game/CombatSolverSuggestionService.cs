using System.Collections;
using System.Reflection;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Nodes;
using STS2AIAgent.Server;

namespace STS2AIAgent.Game;

/// <summary>通过反射调用可选的 CombatSolver，并只导出当前第一步规范动作。</summary>
internal static class CombatSolverSuggestionService
{
    private const string SolverAssemblyName = "CombatSolver";
    private const int SearchTimeoutMilliseconds = 130_000;
    private static readonly SemaphoreSlim SuggestionGate = new(1, 1);

    /// <summary>在当前 live revision 上搜索一次，但不让 Solver 执行任何动作。</summary>
    /// <param name="expectedStateRevision">学生实际观察到的状态 revision。</param>
    /// <param name="cancellationToken">HTTP 服务关闭或请求取消信号。</param>
    /// <returns>只包含规范动作、Solver 版本和绑定 revision 的响应。</returns>
    /// <exception cref="ApiException">Solver 缺失、状态过期、搜索失败或超时。</exception>
    public static async Task<SolverSuggestionPayload> SuggestAsync(
        long expectedStateRevision,
        CancellationToken cancellationToken)
    {
        await SuggestionGate.WaitAsync(cancellationToken);
        try
        {
            GameStatePayload beforeState = GameStateService.BuildStatePayload();
            if (beforeState.state_revision != expectedStateRevision)
            {
                throw new ApiException(
                    409,
                    "stale_state",
                    "Observed state revision is no longer current.",
                    new
                    {
                        expected_state_revision = expectedStateRevision,
                        actual_state_revision = beforeState.state_revision,
                        current_state = beforeState
                    },
                    retryable: true);
            }

            NGame host = NGame.Instance
                ?? throw SolverUnavailable("Game host is unavailable.");
            CombatState state = CombatManager.Instance.DebugOnlyGetState()
                ?? throw new ApiException(
                    409,
                    "solver_state_unavailable",
                    "Current state is not an active combat.");
            Assembly assembly = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(candidate => candidate.GetName().Name == SolverAssemblyName)
                ?? throw SolverUnavailable("CombatSolver is not loaded.");
            Type controller = assembly.GetType(
                    "CombatSolver.SolverController",
                    throwOnError: false)
                ?? throw SolverUnavailable("CombatSolver controller is unavailable.");
            Type reasonType = assembly.GetType(
                    "CombatSolver.SearchReason",
                    throwOnError: false)
                ?? throw SolverUnavailable("CombatSolver search reason is unavailable.");
            MethodInfo requestSearch = controller
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .SingleOrDefault(method =>
                    method.Name == "RequestSearch" && method.GetParameters().Length == 4)
                ?? throw SolverUnavailable("CombatSolver search entry is unavailable.");
            PropertyInfo isSearching = RequiredStaticProperty(controller, "IsSearching");
            PropertyInfo fullAutoEnabled = RequiredStaticProperty(
                controller,
                "FullAutoEnabled");
            PropertyInfo currentResult = RequiredStaticProperty(
                controller,
                "CurrentResultForBugReport");
            PropertyInfo lastFailure = RequiredStaticProperty(
                controller,
                "LastSearchFailureForTesting");
            if (fullAutoEnabled.GetValue(null) is true)
            {
                throw new ApiException(
                    409,
                    "solver_full_auto_enabled",
                    "Disable CombatSolver full auto before requesting a read-only suggestion.");
            }
            object manualReason = Enum.Parse(reasonType, "Manual");
            object? previousResult = currentResult.GetValue(null);
            try
            {
                requestSearch.Invoke(null, [host, state, manualReason, false]);
            }
            catch (TargetInvocationException ex) when (ex.InnerException != null)
            {
                throw new ApiException(
                    422,
                    "solver_search_rejected",
                    ex.InnerException.Message);
            }

            long deadline = System.Environment.TickCount64 + SearchTimeoutMilliseconds;
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                GameStatePayload currentState = GameStateService.BuildStatePayload();
                if (currentState.state_revision != expectedStateRevision)
                {
                    throw new ApiException(
                        409,
                        "solver_state_changed",
                        "Game state changed while CombatSolver was searching.",
                        new
                        {
                            expected_state_revision = expectedStateRevision,
                            actual_state_revision = currentState.state_revision,
                            current_state = currentState
                        },
                        retryable: true);
                }

                if (lastFailure.GetValue(null) is Exception searchFailure)
                {
                    throw new ApiException(
                        422,
                        "solver_search_failed",
                        searchFailure.Message);
                }
                if (fullAutoEnabled.GetValue(null) is true)
                {
                    throw new ApiException(
                        409,
                        "solver_full_auto_enabled",
                        "CombatSolver full auto was enabled during a read-only suggestion.");
                }

                bool searching = isSearching.GetValue(null) is true;
                object? result = currentResult.GetValue(null);
                if (!searching && result != null && !ReferenceEquals(result, previousResult))
                {
                    return new SolverSuggestionPayload
                    {
                        action = FirstAction(result, beforeState),
                        solver_version = SolverVersion(assembly),
                        state_revision = expectedStateRevision
                    };
                }

                if (!searching && result == null)
                {
                    throw new ApiException(
                        422,
                        "solver_search_rejected",
                        "CombatSolver did not start a search for the current state.");
                }
                if (!searching && ReferenceEquals(result, previousResult))
                {
                    throw new ApiException(
                        422,
                        "solver_search_rejected",
                        "CombatSolver kept the previous result instead of starting a new search.");
                }
                if (System.Environment.TickCount64 >= deadline)
                {
                    throw new ApiException(
                        504,
                        "solver_search_timeout",
                        "CombatSolver did not finish within the suggestion timeout.",
                        retryable: true);
                }
                await host.ToSignal(host.GetTree(), SceneTree.SignalName.ProcessFrame);
            }
        }
        finally
        {
            SuggestionGate.Release();
        }
    }

    /// <summary>从 Solver 最佳路线中取得当前回合第一条动作并映射到 Harness 索引。</summary>
    /// <param name="result">CombatSolver 内部的完成结果。</param>
    /// <param name="state">请求搜索时的 Agent 可见状态。</param>
    /// <returns>一行规范 <c>ACTION:</c> 动作。</returns>
    /// <exception cref="ApiException">路线为空或无法映射到当前手牌和药水栏。</exception>
    private static string FirstAction(object result, GameStatePayload state)
    {
        int startTurn = Convert.ToInt32(RequiredProperty(result, "StartTurnNumber"));
        object bestNode = RequiredProperty(result, "BestNode");
        var actions = RequiredProperty(bestNode, "Actions") as IEnumerable
            ?? throw InvalidResult("CombatSolver result does not expose actions.");
        object? first = actions.Cast<object>().FirstOrDefault(action =>
            Convert.ToInt32(RequiredProperty(action, "Turn")) == startTurn);
        if (first == null)
            throw InvalidResult("CombatSolver result has no action for the current turn.");

        string kind = RequiredProperty(first, "Kind").ToString() ?? string.Empty;
        int targetIndex = Convert.ToInt32(RequiredProperty(first, "TargetIndex"));
        string target = targetIndex >= 0 ? $" {targetIndex}" : string.Empty;
        if (kind == "EndTurn")
            return "ACTION: end_turn";
        if (kind == "UsePotion")
        {
            int slot = Convert.ToInt32(RequiredProperty(first, "PotionSlot"));
            return $"ACTION: use_potion {slot}{target}";
        }
        if (kind != "PlayCard")
            throw InvalidResult($"Unsupported CombatSolver action kind: {kind}.");

        string cardId = RequiredProperty(first, "CardId").ToString() ?? string.Empty;
        int occurrence = Convert.ToInt32(RequiredProperty(first, "CardOccurrence"));
        CombatHandCardPayload? card = state.combat?.hand
            .Where(candidate => candidate.card_id == cardId)
            .Skip(occurrence)
            .FirstOrDefault();
        if (card == null)
        {
            throw InvalidResult(
                $"CombatSolver card is absent from the current hand: {cardId}#{occurrence}.");
        }
        return $"ACTION: play_card {card.index}{target}";
    }

    /// <summary>读取内部 Solver 类型的静态属性。</summary>
    /// <param name="type">CombatSolver 内部控制器类型。</param>
    /// <param name="name">属性名。</param>
    /// <returns>可读取的静态属性。</returns>
    /// <exception cref="ApiException">当前 Solver 版本不再提供该属性。</exception>
    private static PropertyInfo RequiredStaticProperty(Type type, string name)
    {
        return type.GetProperty(
                   name,
                   BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
               ?? throw SolverUnavailable(
                   $"CombatSolver property is unavailable: {name}.");
    }

    /// <summary>读取内部 Solver 结果对象的公开或内部实例属性。</summary>
    /// <param name="target">Solver 结果或动作对象。</param>
    /// <param name="name">属性名。</param>
    /// <returns>非空属性值。</returns>
    /// <exception cref="ApiException">属性缺失或为空。</exception>
    private static object RequiredProperty(object target, string name)
    {
        return target.GetType().GetProperty(
                   name,
                   BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                   ?.GetValue(target)
               ?? throw InvalidResult(
                   $"CombatSolver result property is unavailable: {name}.");
    }

    /// <summary>取得实际加载的 Solver 组件版本。</summary>
    /// <param name="assembly">当前加载的 CombatSolver 程序集。</param>
    /// <returns>不包含构建元数据的版本字符串。</returns>
    private static string SolverVersion(Assembly assembly)
    {
        string? informational = assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion;
        return string.IsNullOrWhiteSpace(informational)
            ? assembly.GetName().Version?.ToString() ?? "unknown"
            : informational.Split('+', 2)[0];
    }

    /// <summary>构造可重试的 Solver 未加载错误。</summary>
    /// <param name="message">供调用方诊断的精简原因。</param>
    /// <returns>HTTP 503 API 异常。</returns>
    private static ApiException SolverUnavailable(string message)
    {
        return new ApiException(503, "solver_unavailable", message, retryable: true);
    }

    /// <summary>构造 Solver 完成结果无法映射到 Harness 的错误。</summary>
    /// <param name="message">结果失配原因。</param>
    /// <returns>HTTP 422 API 异常。</returns>
    private static ApiException InvalidResult(string message)
    {
        return new ApiException(422, "solver_result_invalid", message);
    }
}

/// <summary>只向 DAgger 客户端公开规范动作和版本，不公开搜索诊断。</summary>
internal sealed class SolverSuggestionPayload
{
    /// <summary>获取 Solver 建议的 Harness 规范动作。</summary>
    public string action { get; init; } = string.Empty;

    /// <summary>获取当前加载的 CombatSolver 版本。</summary>
    public string solver_version { get; init; } = string.Empty;

    /// <summary>获取建议绑定的状态 revision。</summary>
    public long state_revision { get; init; }
}
