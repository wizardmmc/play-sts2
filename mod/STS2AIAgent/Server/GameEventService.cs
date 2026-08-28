using System.Threading.Channels;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using STS2AIAgent.Game;

namespace STS2AIAgent.Server;

internal sealed class GameEventService
{
    private const string LogPrefix = "[STS2AIAgent.GameEventService]";
    private const int SubscriberBufferCapacity = 64;
    private static readonly Lazy<GameEventService> LazyInstance = new(() => new GameEventService());
    private static readonly IEqualityComparer<RelicModel> RelicReferenceComparer =
        ReferenceEqualityComparer.Instance;

    private readonly object _gate = new();
    private readonly object _stateGate = new();
    private readonly object _lifecycleGate = new();
    private readonly object _bindingGate = new();
    private readonly Dictionary<long, Channel<GameEventEnvelope>> _subscribers = new();
    private readonly List<Action> _managerUnsubscribers = new();

    private bool _started;
    private bool _capturePending;
    private Task? _captureTask;
    private CancellationTokenSource? _captureCancellation;
    private long _lifecycleGeneration;
    private long _captureGeneration;
    private long _nextSubscriberId;
    private long _nextEventId;
    private StateDigest? _lastState;
    private GameStatePayload? _lastStatePayload;
    private ActionQueueSet? _actionQueueSet;
    private ActionExecutor? _actionExecutor;
    private PlayerCombatState? _playerCombatState;
    private CardPile? _hand;
    private Player? _localPlayer;
    private Action? _runInfrastructureUnsubscribe;
    private Action? _actionExecutorUnsubscribe;
    private Action? _combatStateUnsubscribe;
    private Action? _playerStateUnsubscribe;
    private readonly Dictionary<RelicModel, Action> _boundRelics = new(
        RelicReferenceComparer);

    public static GameEventService Instance => LazyInstance.Value;

    private GameEventService() { }

    public void Start()
    {
        lock (_lifecycleGate)
        {
            long generation;
            lock (_gate)
            {
                if (_started)
                {
                    return;
                }

                _started = true;
                generation = ++_lifecycleGeneration;
                _captureCancellation = new CancellationTokenSource();
            }

            try
            {
                BindManagerEvents(generation);
                BindRunInfrastructure(generation);
                NotifyStateChanged();
                Log.Info($"{LogPrefix} Started with event-driven state capture");
            }
            catch (Exception ex)
            {
                List<Channel<GameEventEnvelope>> channels;
                CancellationTokenSource? captureCancellation;
                Task? captureTask;
                lock (_stateGate)
                {
                    lock (_gate)
                    {
                        _started = false;
                        _lifecycleGeneration++;
                        _capturePending = false;
                        captureCancellation = _captureCancellation;
                        _captureCancellation = null;
                        captureTask = _captureTask;
                        _captureTask = null;
                        _lastState = null;
                        _lastStatePayload = null;
                        channels = _subscribers.Values.ToList();
                        _subscribers.Clear();
                    }
                }
                CancelAndObserveCapture(captureCancellation, captureTask);
                UnbindManagerEvents();
                UnbindRunInfrastructure();
                UnbindCombatState();
                UnbindPlayerState();
                TryObserverCleanup(
                    GameStateService.MarkCombatTurnClosed,
                    "close combat turn after failed start");
                foreach (var channel in channels)
                {
                    channel.Writer.TryComplete();
                }
                Log.Error($"{LogPrefix} Failed to start: {ex}");
                throw;
            }
        }
    }

    public void Stop()
    {
        lock (_lifecycleGate)
        {
            List<Channel<GameEventEnvelope>> channels;
            CancellationTokenSource? captureCancellation;
            Task? captureTask;

            lock (_stateGate)
            {
                lock (_gate)
                {
                    if (!_started)
                    {
                        return;
                    }

                    _started = false;
                    _lifecycleGeneration++;
                    _capturePending = false;
                    captureCancellation = _captureCancellation;
                    _captureCancellation = null;
                    captureTask = _captureTask;
                    _captureTask = null;
                    _lastState = null;
                    _lastStatePayload = null;
                    channels = _subscribers.Values.ToList();
                    _subscribers.Clear();
                }
            }
            CancelAndObserveCapture(captureCancellation, captureTask);

            try
            {
                UnbindManagerEvents();
                UnbindRunInfrastructure();
                UnbindCombatState();
                UnbindPlayerState();
                TryObserverCleanup(
                    GameStateService.MarkCombatTurnClosed,
                    "close combat turn during stop");
            }
            finally
            {
                foreach (var channel in channels)
                {
                    channel.Writer.TryComplete();
                }

                Log.Info($"{LogPrefix} Stopped");
            }
        }
    }

    private void BindManagerEvents(long generation)
    {
        void RunStarted(RunState state) => RunLifecycleCallback(
            generation, () => OnRunStarted(state, generation));
        void RoomChanged() => RunLifecycleCallback(
            generation, () => OnRoomSignal(generation));
        void ActiveScreenChanged() => RunLifecycleCallback(
            generation, OnStateSignal);
        void CombatSetUp(CombatState state) => RunLifecycleCallback(
            generation, () => OnCombatSetUp(state, generation));
        void CombatEnded(CombatRoom room) => RunLifecycleCallback(
            generation, () => OnCombatEnded(room, generation));
        void CombatStateChanged(CombatState state) => RunLifecycleCallback(
            generation, () => OnCombatStateChanged(state, generation));
        void TurnStarted(CombatState state) => RunLifecycleCallback(
            generation, () => OnTurnStarted(state, generation));
        void TurnEnded(CombatState state) => RunLifecycleCallback(
            generation, () => OnTurnEnded(state));
        void PlayerEndedTurn(Player player, bool forced) => RunLifecycleCallback(
            generation, () => OnPlayerEndedTurn(player, forced));
        void PlayerUnendedTurn(Player player) => RunLifecycleCallback(
            generation, () => OnPlayerUnendedTurn(player));

        void Subscribe(Action add, Action remove)
        {
            _managerUnsubscribers.Add(remove);
            add();
        }

        Subscribe(
            () => RunManager.Instance.RunStarted += RunStarted,
            () => RunManager.Instance.RunStarted -= RunStarted);
        Subscribe(
            () => RunManager.Instance.RoomEntered += RoomChanged,
            () => RunManager.Instance.RoomEntered -= RoomChanged);
        Subscribe(
            () => RunManager.Instance.RoomExited += RoomChanged,
            () => RunManager.Instance.RoomExited -= RoomChanged);
        Subscribe(
            () => ActiveScreenContext.Instance.Updated += ActiveScreenChanged,
            () => ActiveScreenContext.Instance.Updated -= ActiveScreenChanged);
        Subscribe(
            () => CombatManager.Instance.CombatSetUp += CombatSetUp,
            () => CombatManager.Instance.CombatSetUp -= CombatSetUp);
        Subscribe(
            () => CombatManager.Instance.CombatEnded += CombatEnded,
            () => CombatManager.Instance.CombatEnded -= CombatEnded);
        Subscribe(
            () => CombatManager.Instance.CreaturesChanged += CombatStateChanged,
            () => CombatManager.Instance.CreaturesChanged -= CombatStateChanged);
        Subscribe(
            () => CombatManager.Instance.TurnStarted += TurnStarted,
            () => CombatManager.Instance.TurnStarted -= TurnStarted);
        Subscribe(
            () => CombatManager.Instance.TurnEnded += TurnEnded,
            () => CombatManager.Instance.TurnEnded -= TurnEnded);
        Subscribe(
            () => CombatManager.Instance.PlayerEndedTurn += PlayerEndedTurn,
            () => CombatManager.Instance.PlayerEndedTurn -= PlayerEndedTurn);
        Subscribe(
            () => CombatManager.Instance.PlayerUnendedTurn += PlayerUnendedTurn,
            () => CombatManager.Instance.PlayerUnendedTurn -= PlayerUnendedTurn);
        Subscribe(
            () => CombatManager.Instance.AboutToSwitchToEnemyTurn += TurnEnded,
            () => CombatManager.Instance.AboutToSwitchToEnemyTurn -= TurnEnded);
        Subscribe(
            () => CombatManager.Instance.PlayerActionsDisabledChanged += CombatStateChanged,
            () => CombatManager.Instance.PlayerActionsDisabledChanged -= CombatStateChanged);
    }

    private void UnbindManagerEvents()
    {
        var unsubscribers = _managerUnsubscribers.ToArray();
        _managerUnsubscribers.Clear();
        foreach (var unsubscribe in unsubscribers)
        {
            TryObserverCleanup(unsubscribe, "unbind manager events");
        }
    }

    public long CaptureGeneration()
    {
        lock (_gate)
        {
            if (!_started)
            {
                throw new InvalidOperationException(
                    "Game event service is not running.");
            }
            return _lifecycleGeneration;
        }
    }

    public GameEventSubscription Subscribe(
        GameStatePayload snapshot,
        long generation)
    {
        var channel = Channel.CreateBounded<GameEventEnvelope>(new BoundedChannelOptions(
            SubscriberBufferCapacity)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.Wait
        });

        long subscriberId;
        GameStatePayload readySnapshot;
        lock (_stateGate)
        {
            lock (_gate)
            {
                if (!_started || generation != _lifecycleGeneration)
                {
                    throw new InvalidOperationException(
                        "Game event service lifecycle changed while opening the stream.");
                }
            }

            ProcessState(snapshot);
            readySnapshot = _lastStatePayload ?? snapshot;
            lock (_gate)
            {
                subscriberId = ++_nextSubscriberId;
                _subscribers[subscriberId] = channel;
            }

            var digest = StateDigest.FromState(readySnapshot);
            var streamReady = BuildEnvelope("stream_ready", new
            {
                state = readySnapshot,
                run_id = digest.RunId,
                screen = digest.Screen,
                session_phase = digest.SessionPhase,
                character_id = digest.CharacterId,
                ascension = digest.Ascension,
                in_combat = digest.InCombat,
                turn = digest.Turn,
                action_window_open = digest.PlayerActionWindowOpen
            });
            channel.Writer.TryWrite(streamReady);
        }
        NotifyStateChanged();

        return new GameEventSubscription(subscriberId, channel.Reader, Unsubscribe);
    }

    private void Unsubscribe(long subscriberId)
    {
        Channel<GameEventEnvelope>? channel = null;
        lock (_gate)
        {
            if (_subscribers.Remove(subscriberId, out var removed))
            {
                channel = removed;
            }
        }

        channel?.Writer.TryComplete();
    }

    public void ObserveState(GameStatePayload state, long generation)
    {
        lock (_stateGate)
        {
            lock (_gate)
            {
                if (!_started || generation != _lifecycleGeneration)
                {
                    return;
                }
            }

            ProcessState(state);
        }
    }

    public void NotifyStateChanged()
    {
        lock (_gate)
        {
            if (!_started)
            {
                return;
            }

            _capturePending = true;
            if (_captureTask == null)
            {
                var captureCancellation = _captureCancellation;
                if (captureCancellation == null)
                {
                    return;
                }
                _captureGeneration = _lifecycleGeneration;
                _captureTask = CaptureChangedStatesAsync(
                    _captureGeneration,
                    captureCancellation.Token);
            }
        }
    }

    private async Task CaptureChangedStatesAsync(
        long generation,
        CancellationToken cancellationToken)
    {
        await Task.Yield();
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            lock (_gate)
            {
                if (!_started || generation != _lifecycleGeneration)
                {
                    return;
                }
                if (!_capturePending)
                {
                    if (_captureGeneration == generation)
                    {
                        _captureTask = null;
                    }
                    return;
                }

                _capturePending = false;
            }

            try
            {
                var state = await GameThread.InvokeAsync(async () =>
                {
                    await WaitForNextFrameAsync(cancellationToken);
                    cancellationToken.ThrowIfCancellationRequested();
                    return GameStateService.BuildStatePayload();
                });
                ObserveCapturedState(state, generation);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                return;
            }
            catch (Exception ex)
            {
                Log.Warn($"{LogPrefix} Event-driven state capture failed: {ex.Message}");
            }
        }
    }

    private void ObserveCapturedState(
        GameStatePayload state,
        long generation)
    {
        lock (_stateGate)
        {
            lock (_gate)
            {
                if (!_started || generation != _lifecycleGeneration)
                {
                    return;
                }
            }

            ProcessState(state);
        }
    }

    private void RunLifecycleCallback(long generation, Action callback)
    {
        lock (_lifecycleGate)
        {
            lock (_gate)
            {
                if (!_started || generation != _lifecycleGeneration)
                {
                    return;
                }
            }

            try
            {
                callback();
            }
            catch (Exception ex)
            {
                Log.Warn($"{LogPrefix} Ignored observer callback failure: {ex}");
            }
        }
    }

    private static void TryObserverCleanup(Action cleanup, string operation)
    {
        try
        {
            cleanup();
        }
        catch (Exception ex)
        {
            Log.Warn($"{LogPrefix} Failed to {operation}: {ex}");
        }
    }

    private static void CancelAndObserveCapture(
        CancellationTokenSource? cancellation,
        Task? captureTask)
    {
        if (cancellation == null)
        {
            return;
        }

        try
        {
            cancellation.Cancel();
        }
        catch (Exception ex)
        {
            Log.Warn($"{LogPrefix} Failed to cancel state capture: {ex}");
        }

        if (captureTask == null)
        {
            cancellation.Dispose();
            return;
        }

        _ = ObserveStoppedCaptureAsync(captureTask, cancellation);
    }

    private static async Task ObserveStoppedCaptureAsync(
        Task captureTask,
        CancellationTokenSource cancellation)
    {
        try
        {
            await captureTask.ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            Log.Warn($"{LogPrefix} State capture failed while stopping: {ex}");
        }
        finally
        {
            cancellation.Dispose();
        }
    }

    private void OnRunStarted(RunState _, long generation)
    {
        BindRunInfrastructure(generation);
        BindCurrentRunPlayer(generation);
        NotifyStateChanged();
    }

    private void OnCombatSetUp(CombatState state, long generation)
    {
        GameStateService.MarkCombatTurnClosed();
        BindCombatState(state, generation);
        NotifyStateChanged();
    }

    private void OnCombatEnded(CombatRoom _, long generation)
    {
        GameStateService.MarkCombatTurnClosed();
        UnbindCombatState();
        BindCurrentRunPlayer(generation);
        NotifyStateChanged();
    }

    private void OnTurnStarted(CombatState state, long generation)
    {
        GameStateService.MarkCombatTurnStarted(state);
        BindCombatState(state, generation);
        NotifyStateChanged();
    }

    private void OnTurnEnded(CombatState _)
    {
        GameStateService.MarkCombatTurnClosed();
        NotifyStateChanged();
    }

    private void OnPlayerEndedTurn(Player _, bool __)
    {
        GameStateService.MarkCombatTurnClosed();
        NotifyStateChanged();
    }

    private void OnPlayerUnendedTurn(Player _)
    {
        var state = CombatManager.Instance.DebugOnlyGetState();
        if (state != null)
        {
            GameStateService.MarkCombatTurnStarted(state);
        }
        NotifyStateChanged();
    }

    private void OnCombatStateChanged(CombatState state, long generation)
    {
        BindCombatState(state, generation);
        NotifyStateChanged();
    }

    private void OnPlayerTurnPhaseChanged()
    {
        if (_playerCombatState?.Phase != PlayerTurnPhase.Play)
        {
            GameStateService.MarkCombatTurnClosed();
        }
        NotifyStateChanged();
    }

    private void OnNumericStateChanged(int _, int __) => NotifyStateChanged();

    private void OnGameActionChanged(GameAction _) => NotifyStateChanged();

    private void OnStateSignal() => NotifyStateChanged();

    private void OnRoomSignal(long generation)
    {
        BindCurrentRunPlayer(generation);
        NotifyStateChanged();
    }

    private void BindRunInfrastructure(long generation)
    {
        var queueSet = RunManager.Instance.ActionQueueSet;
        if (!ReferenceEquals(queueSet, _actionQueueSet))
        {
            var previousQueueUnsubscribe = _runInfrastructureUnsubscribe;
            _runInfrastructureUnsubscribe = null;
            if (previousQueueUnsubscribe != null)
            {
                TryObserverCleanup(
                    previousQueueUnsubscribe,
                    "replace action queue events");
            }
            _actionQueueSet = queueSet;
            void QueueChanged() => RunLifecycleCallback(
                generation, OnStateSignal);
            void ActionEnqueued(GameAction action) => RunLifecycleCallback(
                generation, () => OnGameActionChanged(action));
            _runInfrastructureUnsubscribe = () =>
            {
                TryObserverCleanup(
                    () => queueSet.ActionQueueChanged -= QueueChanged,
                    "unbind action queue changed event");
                TryObserverCleanup(
                    () => queueSet.ActionEnqueued -= ActionEnqueued,
                    "unbind action enqueued event");
            };
            _actionQueueSet.ActionQueueChanged += QueueChanged;
            _actionQueueSet.ActionEnqueued += ActionEnqueued;
        }

        var executor = RunManager.Instance.ActionExecutor;
        if (!ReferenceEquals(executor, _actionExecutor))
        {
            var previousExecutorUnsubscribe = _actionExecutorUnsubscribe;
            _actionExecutorUnsubscribe = null;
            if (previousExecutorUnsubscribe != null)
            {
                TryObserverCleanup(
                    previousExecutorUnsubscribe,
                    "replace action executor events");
            }
            _actionExecutor = executor;
            void ActionExecuted(GameAction action) => RunLifecycleCallback(
                generation, () => OnGameActionChanged(action));
            _actionExecutorUnsubscribe = () =>
                executor.AfterActionExecuted -= ActionExecuted;
            _actionExecutor.AfterActionExecuted += ActionExecuted;
        }

        BindCurrentRunPlayer(generation);
    }

    private void UnbindRunInfrastructure()
    {
        var queueUnsubscribe = _runInfrastructureUnsubscribe;
        _runInfrastructureUnsubscribe = null;
        _actionQueueSet = null;
        if (queueUnsubscribe != null)
        {
            TryObserverCleanup(queueUnsubscribe, "unbind action queue events");
        }

        var executorUnsubscribe = _actionExecutorUnsubscribe;
        _actionExecutorUnsubscribe = null;
        _actionExecutor = null;
        if (executorUnsubscribe != null)
        {
            TryObserverCleanup(executorUnsubscribe, "unbind action executor events");
        }
    }

    private void BindCombatState(CombatState state, long generation)
    {
        lock (_bindingGate)
        {
            Player? player;
            try
            {
                player = GameStateService.GetLocalPlayer(state);
            }
            catch (InvalidOperationException)
            {
                // CombatManager.Reset 会先移除本地玩家，再发布 CreaturesChanged。
                // 这是正常的 teardown 窗口，事件观察器不能让异常反向中断清场动作。
                UnbindCombatState();
                return;
            }

            BindPlayerState(player, generation);
            var playerCombatState = player?.PlayerCombatState;

            if (ReferenceEquals(playerCombatState, _playerCombatState))
            {
                return;
            }

            UnbindCombatState();
            _playerCombatState = playerCombatState;
            if (_playerCombatState == null)
            {
                return;
            }
            var boundCombatState = _playerCombatState;

            void TurnPhaseChanged() => RunLifecycleCallback(generation, () =>
            {
                if (ReferenceEquals(_playerCombatState, playerCombatState))
                {
                    OnPlayerTurnPhaseChanged();
                }
            });
            void NumericStateChanged(int before, int after) =>
                RunLifecycleCallback(generation, () =>
                {
                    if (ReferenceEquals(_playerCombatState, playerCombatState))
                    {
                        OnNumericStateChanged(before, after);
                    }
                });
            _hand = boundCombatState.Hand;
            var hand = _hand;
            void HandChanged() => RunLifecycleCallback(generation, () =>
            {
                if (ReferenceEquals(_hand, hand))
                {
                    OnStateSignal();
                }
            });
            _combatStateUnsubscribe = () =>
            {
                TryObserverCleanup(
                    () => boundCombatState.PlayerTurnPhaseChanged -= TurnPhaseChanged,
                    "unbind turn phase event");
                TryObserverCleanup(
                    () => boundCombatState.EnergyChanged -= NumericStateChanged,
                    "unbind combat energy event");
                TryObserverCleanup(
                    () => boundCombatState.StarsChanged -= NumericStateChanged,
                    "unbind combat stars event");
                TryObserverCleanup(
                    () => hand.ContentsChanged -= HandChanged,
                    "unbind hand event");
            };
            boundCombatState.PlayerTurnPhaseChanged += TurnPhaseChanged;
            boundCombatState.EnergyChanged += NumericStateChanged;
            boundCombatState.StarsChanged += NumericStateChanged;
            hand.ContentsChanged += HandChanged;
        }
    }

    private void BindCurrentRunPlayer(long generation)
    {
        Player? player;
        try
        {
            player = GameStateService.GetLocalPlayer(
                RunManager.Instance.DebugOnlyGetState());
        }
        catch (InvalidOperationException)
        {
            player = null;
        }

        BindPlayerState(player, generation);
    }

    private void BindPlayerState(Player? player, long generation)
    {
        lock (_bindingGate)
        {
            if (!ReferenceEquals(player, _localPlayer))
            {
                UnbindPlayerState();
                _localPlayer = player;
                if (_localPlayer == null)
                {
                    return;
                }

                var boundPlayer = _localPlayer;
                void PlayerStateChanged() => RunLifecycleCallback(
                    generation, () =>
                    {
                        if (ReferenceEquals(_localPlayer, boundPlayer))
                        {
                            OnStateSignal();
                        }
                    });
                void IntegerStateChanged(int value) => RunLifecycleCallback(
                    generation, () =>
                    {
                        if (ReferenceEquals(_localPlayer, boundPlayer))
                        {
                            OnIntegerStateChanged(value);
                        }
                    });
                void PotionStateChanged(PotionModel potion) =>
                    RunLifecycleCallback(generation, () =>
                    {
                        if (ReferenceEquals(_localPlayer, boundPlayer))
                        {
                            OnPotionStateChanged(potion);
                        }
                    });
                void RelicObtained(RelicModel relic) => RunLifecycleCallback(
                    generation, () =>
                    {
                        if (!ReferenceEquals(_localPlayer, boundPlayer))
                        {
                            return;
                        }
                        BindRelic(relic, generation);
                        NotifyStateChanged();
                    });
                void RelicRemoved(RelicModel relic) => RunLifecycleCallback(
                    generation, () =>
                    {
                        if (!ReferenceEquals(_localPlayer, boundPlayer))
                        {
                            return;
                        }
                        UnbindRelic(relic);
                        NotifyStateChanged();
                    });

                _playerStateUnsubscribe = () =>
                {
                    TryObserverCleanup(
                        () => boundPlayer.Deck.ContentsChanged -= PlayerStateChanged,
                        "unbind deck event");
                    TryObserverCleanup(
                        () => boundPlayer.RelicObtained -= RelicObtained,
                        "unbind relic obtained event");
                    TryObserverCleanup(
                        () => boundPlayer.RelicRemoved -= RelicRemoved,
                        "unbind relic removed event");
                    TryObserverCleanup(
                        () => boundPlayer.MaxPotionCountChanged -= IntegerStateChanged,
                        "unbind potion capacity event");
                    TryObserverCleanup(
                        () => boundPlayer.PotionProcured -= PotionStateChanged,
                        "unbind potion procured event");
                    TryObserverCleanup(
                        () => boundPlayer.PotionDiscarded -= PotionStateChanged,
                        "unbind potion discarded event");
                    TryObserverCleanup(
                        () => boundPlayer.UsedPotionRemoved -= PotionStateChanged,
                        "unbind potion used event");
                    TryObserverCleanup(
                        () => boundPlayer.GoldChanged -= PlayerStateChanged,
                        "unbind gold event");
                    TryObserverCleanup(
                        () => boundPlayer.CanRemovePotionsChanged -= PlayerStateChanged,
                        "unbind potion removal event");
                };
                boundPlayer.Deck.ContentsChanged += PlayerStateChanged;
                boundPlayer.RelicObtained += RelicObtained;
                boundPlayer.RelicRemoved += RelicRemoved;
                boundPlayer.MaxPotionCountChanged += IntegerStateChanged;
                boundPlayer.PotionProcured += PotionStateChanged;
                boundPlayer.PotionDiscarded += PotionStateChanged;
                boundPlayer.UsedPotionRemoved += PotionStateChanged;
                boundPlayer.GoldChanged += PlayerStateChanged;
                boundPlayer.CanRemovePotionsChanged += PlayerStateChanged;
            }

            RefreshRelicBindings(generation);
        }
    }

    private void UnbindPlayerState()
    {
        lock (_bindingGate)
        {
            var playerUnsubscribe = _playerStateUnsubscribe;
            _playerStateUnsubscribe = null;
            _localPlayer = null;
            if (playerUnsubscribe != null)
            {
                TryObserverCleanup(playerUnsubscribe, "unbind player events");
            }

            foreach (var relic in _boundRelics.Keys.ToArray())
            {
                UnbindRelic(relic);
            }
        }
    }

    private void RefreshRelicBindings(long generation)
    {
        lock (_bindingGate)
        {
            if (_localPlayer == null)
            {
                return;
            }

            var currentRelics = _localPlayer.Relics.ToHashSet(
                RelicReferenceComparer);
            foreach (var relic in _boundRelics.Keys.Where(
                         relic => !currentRelics.Contains(relic)).ToArray())
            {
                UnbindRelic(relic);
            }
            foreach (var relic in currentRelics)
            {
                BindRelic(relic, generation);
            }
        }
    }

    private void BindRelic(RelicModel relic, long generation)
    {
        lock (_bindingGate)
        {
            if (_boundRelics.ContainsKey(relic))
            {
                return;
            }

            void RelicChanged() => RunLifecycleCallback(generation, () =>
            {
                if (_boundRelics.ContainsKey(relic))
                {
                    OnStateSignal();
                }
            });
            _boundRelics.Add(relic, () =>
            {
                TryObserverCleanup(
                    () => relic.DisplayAmountChanged -= RelicChanged,
                    "unbind relic display amount event");
                TryObserverCleanup(
                    () => relic.StatusChanged -= RelicChanged,
                    "unbind relic status event");
            });
            relic.DisplayAmountChanged += RelicChanged;
            relic.StatusChanged += RelicChanged;
        }
    }

    private void UnbindRelic(RelicModel relic)
    {
        lock (_bindingGate)
        {
            if (!_boundRelics.Remove(relic, out var unsubscribe))
            {
                return;
            }

            TryObserverCleanup(unsubscribe, "unbind relic events");
        }
    }

    private void OnIntegerStateChanged(int _) => NotifyStateChanged();

    private void OnPotionStateChanged(PotionModel _) => NotifyStateChanged();

    private void UnbindCombatState()
    {
        lock (_bindingGate)
        {
            var combatUnsubscribe = _combatStateUnsubscribe;
            _combatStateUnsubscribe = null;
            _playerCombatState = null;
            _hand = null;
            if (combatUnsubscribe != null)
            {
                TryObserverCleanup(combatUnsubscribe, "unbind combat events");
            }
        }
    }

    private static async Task WaitForNextFrameAsync(
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var game = NGame.Instance;
        if (game == null || !GodotObject.IsInstanceValid(game))
        {
            await Task.Yield();
            return;
        }

        var tree = game.GetTree();
        if (tree == null || !GodotObject.IsInstanceValid(tree))
        {
            await Task.Yield();
            return;
        }

        var completion = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        CancellationTokenRegistration cancellationRegistration = default;
        var completed = 0;

        void DetachFrameHandlerOnGameThread()
        {
            if (GodotObject.IsInstanceValid(tree))
            {
                tree.ProcessFrame -= OnProcessFrame;
            }
        }

        void OnProcessFrame()
        {
            if (Interlocked.Exchange(ref completed, 1) != 0)
            {
                return;
            }

            try
            {
                DetachFrameHandlerOnGameThread();
                completion.TrySetResult();
            }
            catch (Exception ex)
            {
                completion.TrySetException(ex);
            }
        }

        async Task CancelOnGameThreadAsync()
        {
            try
            {
                var cancellationWon = await GameThread.InvokeAsync(() =>
                {
                    if (Interlocked.Exchange(ref completed, 1) != 0)
                    {
                        return false;
                    }

                    DetachFrameHandlerOnGameThread();
                    return true;
                }).ConfigureAwait(false);
                if (cancellationWon)
                {
                    completion.TrySetCanceled(cancellationToken);
                }
            }
            catch (Exception ex)
            {
                completion.TrySetException(ex);
            }
        }

        tree.ProcessFrame += OnProcessFrame;
        if (cancellationToken.CanBeCanceled)
        {
            cancellationRegistration = cancellationToken.Register(
                () => _ = CancelOnGameThreadAsync());
        }

        try
        {
            await completion.Task.ConfigureAwait(false);
        }
        finally
        {
            cancellationRegistration.Dispose();
        }
    }

    private void ProcessState(GameStatePayload state)
    {
        var current = StateDigest.FromState(state);
        var previous = _lastState;

        if (previous != null && current.StateRevision < previous.StateRevision)
        {
            return;
        }

        if (previous != null && current.StateRevision == previous.StateRevision)
        {
            _lastStatePayload = state;
            return;
        }

        if (previous == null)
        {
            Publish("session_started", new
            {
                run_id = current.RunId,
                screen = current.Screen,
                session_phase = current.SessionPhase
            });
            if (string.Equals(current.SessionPhase, "run", StringComparison.Ordinal))
            {
                Publish("run_started", new
                {
                    run_id = current.RunId,
                    character_id = current.CharacterId,
                    ascension = current.Ascension
                });
            }
            _lastState = current;
            _lastStatePayload = state;
            return;
        }

        if (previous.StateRevision != current.StateRevision)
        {
            Publish("state_changed", new
            {
                state
            });
        }

        if (!string.Equals(previous.Screen, current.Screen, StringComparison.Ordinal))
        {
            Publish("screen_changed", new
            {
                from = previous.Screen,
                to = current.Screen,
                run_id = current.RunId
            });
        }

        if (!string.Equals(previous.SessionPhase, "run", StringComparison.Ordinal) &&
            string.Equals(current.SessionPhase, "run", StringComparison.Ordinal))
        {
            Publish("run_started", new
            {
                run_id = current.RunId,
                character_id = current.CharacterId,
                ascension = current.Ascension
            });
        }

        if (!string.Equals(previous.Screen, "GAME_OVER", StringComparison.Ordinal) &&
            string.Equals(current.Screen, "GAME_OVER", StringComparison.Ordinal))
        {
            Publish("run_ended", new
            {
                run_id = current.RunId,
                reason = "game_over",
                victory = current.IsVictory
            });
        }
        else if (string.Equals(previous.SessionPhase, "run", StringComparison.Ordinal) &&
                 !string.Equals(current.SessionPhase, "run", StringComparison.Ordinal) &&
                 !string.Equals(previous.Screen, "GAME_OVER", StringComparison.Ordinal))
        {
            Publish("run_ended", new
            {
                run_id = previous.RunId,
                reason = "returned_to_menu",
                victory = (bool?)null
            });
        }

        if (!previous.InCombat && current.InCombat)
        {
            Publish("combat_started", new
            {
                run_id = current.RunId,
                turn = current.Turn
            });
        }
        else if (previous.InCombat && !current.InCombat)
        {
            Publish("combat_ended", new
            {
                run_id = current.RunId
            });
        }
        else if (current.InCombat && previous.Turn != current.Turn)
        {
            Publish("combat_turn_changed", new
            {
                run_id = current.RunId,
                from = previous.Turn,
                to = current.Turn
            });
        }

        if (previous.PlayerActionWindowOpen != current.PlayerActionWindowOpen)
        {
            Publish(current.PlayerActionWindowOpen ? "player_action_window_opened" : "player_action_window_closed", new
            {
                run_id = current.RunId,
                screen = current.Screen,
                actions = current.AvailableActions
            });
        }

        if (!previous.RouteDecisionRequired && current.RouteDecisionRequired)
        {
            Publish("route_decision_required", new
            {
                run_id = current.RunId,
                screen = current.Screen,
                available_nodes = current.AvailableMapNodes
            });
        }

        if (!previous.RewardDecisionRequired && current.RewardDecisionRequired)
        {
            Publish("reward_decision_required", new
            {
                run_id = current.RunId,
                screen = current.Screen,
                reward_count = current.RewardCount,
                card_option_count = current.RewardCardOptionCount
            });
        }

        if (!string.Equals(previous.EventId, current.EventId, StringComparison.Ordinal) ||
            previous.EventOptionCount != current.EventOptionCount ||
            previous.EventFinished != current.EventFinished)
        {
            if (!string.IsNullOrEmpty(current.EventId) || current.EventOptionCount > 0)
            {
                Publish("event_state_changed", new
                {
                    run_id = current.RunId,
                    event_id = current.EventId,
                    option_count = current.EventOptionCount,
                    is_finished = current.EventFinished
                });
            }
        }

        if (!string.Equals(previous.ActionSignature, current.ActionSignature, StringComparison.Ordinal))
        {
            Publish("available_actions_changed", new
            {
                run_id = current.RunId,
                screen = current.Screen,
                actions = current.AvailableActions
            });
        }

        _lastState = current;
        _lastStatePayload = state;
    }

    public void PublishActionExecuted(
        ActionRequest request,
        GameStatePayload beforeState,
        ActionResponsePayload response,
        long generation)
    {
        RunLifecycleCallback(generation, () =>
        {
            Publish("action_executed", new
            {
                request = new
                {
                    action = request.action,
                    expected_state_revision = request.expected_state_revision,
                    card_index = request.card_index,
                    target_index = request.target_index,
                    option_index = request.option_index,
                    command = request.command,
                    client_context = request.client_context
                },
                before_state = beforeState,
                after_state = response.state,
                status = response.status,
                stable = response.stable
            });
            ObserveState(response.state, generation);
            NotifyStateChanged();
        });
    }

    public void PublishNativeUiCaptureGap(
        string action,
        string reason,
        long generation)
    {
        RunLifecycleCallback(generation, () =>
            Publish("native_ui_capture_gap", new
            {
                action,
                reason
            }));
    }

    private void Publish(string eventType, object data)
    {
        var envelope = BuildEnvelope(eventType, data);
        List<long> staleSubscriberIds = new();

        lock (_gate)
        {
            foreach (var (subscriberId, channel) in _subscribers)
            {
                if (!channel.Writer.TryWrite(envelope))
                {
                    staleSubscriberIds.Add(subscriberId);
                }
            }

            foreach (var subscriberId in staleSubscriberIds)
            {
                if (_subscribers.Remove(subscriberId, out var stale))
                {
                    stale.Writer.TryComplete();
                }
            }
        }
    }

    private GameEventEnvelope BuildEnvelope(string eventType, object data)
    {
        var eventId = Interlocked.Increment(ref _nextEventId);
        return new GameEventEnvelope
        {
            event_id = eventId,
            type = eventType,
            timestamp_utc = DateTime.UtcNow.ToString("O"),
            data = data
        };
    }

    private sealed class StateDigest
    {
        public string RunId { get; init; } = "run_unknown";
        public long StateRevision { get; init; }
        public string Screen { get; init; } = "UNKNOWN";
        public string SessionPhase { get; init; } = "menu";
        public string? CharacterId { get; init; }
        public int? Ascension { get; init; }
        public bool? IsVictory { get; init; }
        public bool InCombat { get; init; }
        public int? Turn { get; init; }
        public string[] AvailableActions { get; init; } = Array.Empty<string>();
        public string ActionSignature { get; init; } = string.Empty;
        public bool PlayerActionWindowOpen { get; init; }
        public bool RouteDecisionRequired { get; init; }
        public int AvailableMapNodes { get; init; }
        public bool RewardDecisionRequired { get; init; }
        public int RewardCount { get; init; }
        public int RewardCardOptionCount { get; init; }
        public string EventId { get; init; } = string.Empty;
        public int EventOptionCount { get; init; }
        public bool EventFinished { get; init; }

        public static StateDigest FromState(GameStatePayload state)
        {
            var actions = (state.available_actions ?? Array.Empty<string>())
                .Where(static action => !string.IsNullOrWhiteSpace(action))
                .Distinct(StringComparer.Ordinal)
                .OrderBy(static action => action, StringComparer.Ordinal)
                .ToArray();

            var actionSet = new HashSet<string>(actions, StringComparer.Ordinal);
            var actionWindowOpen = actionSet.Contains("play_card") ||
                                   actionSet.Contains("end_turn") ||
                                   actionSet.Contains("confirm_selection");
            var routeDecisionRequired = actionSet.Contains("choose_map_node");
            var rewardDecisionRequired = state.screen == "REWARD" ||
                                         actionSet.Contains("collect_rewards_and_proceed") ||
                                         actionSet.Contains("claim_reward") ||
                                         actionSet.Contains("choose_reward_card");

            return new StateDigest
            {
                RunId = state.run_id,
                StateRevision = state.state_revision,
                Screen = state.screen,
                SessionPhase = state.session.phase,
                CharacterId = state.run?.character_id,
                Ascension = state.run?.ascension,
                IsVictory = state.game_over?.is_victory,
                InCombat = state.in_combat,
                Turn = state.turn,
                AvailableActions = actions,
                ActionSignature = string.Join("|", actions),
                PlayerActionWindowOpen = actionWindowOpen,
                RouteDecisionRequired = routeDecisionRequired,
                AvailableMapNodes = state.map?.available_nodes?.Length ?? 0,
                RewardDecisionRequired = rewardDecisionRequired,
                RewardCount = state.reward?.rewards?.Length ?? 0,
                RewardCardOptionCount = state.reward?.card_options?.Length ?? 0,
                EventId = state.@event?.event_id ?? string.Empty,
                EventOptionCount = state.@event?.options?.Length ?? 0,
                EventFinished = state.@event?.is_finished ?? false
            };
        }
    }
}

internal sealed class GameEventSubscription : IDisposable
{
    private readonly Action<long> _onDispose;
    private readonly long _subscriberId;
    private int _disposed;

    public GameEventSubscription(long subscriberId, ChannelReader<GameEventEnvelope> reader, Action<long> onDispose)
    {
        _subscriberId = subscriberId;
        _onDispose = onDispose;
        Reader = reader;
    }

    public ChannelReader<GameEventEnvelope> Reader { get; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _onDispose(_subscriberId);
    }
}

internal sealed class GameEventEnvelope
{
    public long event_id { get; init; }

    public string type { get; init; } = string.Empty;

    public string timestamp_utc { get; init; } = string.Empty;

    public object? data { get; init; }
}
