using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Rewards;
using STS2AIAgent.Server;

namespace STS2AIAgent.Game;

/// <summary>只观察原生 UI 决策，不主动驱动游戏。</summary>
internal static class NativeUiActionRecorder
{
    private const string HumanUiSource = "human_ui";
    private const string CombatSolverSource = "combat_solver";
    private const string CombatSolverAssemblyName = "CombatSolver";
    private static Harmony? _harmony;
    private static int _suppressionDepth;
    private static Capture? _pendingCardCapture;
    private static CardModel? _pendingCard;
    private static Capture? _pendingPotionCapture;
    private static PotionModel? _pendingPotion;
    private static Capture? _pendingRewardCapture;
    private static Reward? _pendingReward;
    private static NCardGridSelectionScreen? _pendingGridScreen;
    private static readonly List<(CardModel Card, Capture Capture)>
        _pendingGridSelections = new();
    private static CombatPileSelectionCapture? _activeCombatPileSelection;
    private static string _lastCombatActionSource = HumanUiSource;
    private static long _lastCombatActionGeneration = -1;
    [ThreadStatic] private static int _buttonPatchDepth;
    [ThreadStatic] private static int _cardUiCommitDepth;

    /// <summary>保存动作提交前的请求、状态、局代次和执行者。</summary>
    internal sealed record Capture(
        ActionRequest Request,
        GameStatePayload Before,
        long LifecycleGeneration,
        string ActionSource);
    /// <summary>保存原生提交点预期发布的动作及其执行者。</summary>
    private sealed record CommitCapture(Capture? Capture, string ActionSource);
    private sealed record NativeAttempt(string Action, Capture? Capture);
    private sealed record GridCommitCapture(
        Capture[] Selections, NativeAttempt? Confirmation);
    /// <summary>保存一次原版战斗牌堆选择的候选、来源和可见点击状态。</summary>
    /// <param name="Options">原版过滤后的候选牌。</param>
    /// <param name="Prefs">原版选择数量与确认规则。</param>
    /// <param name="LifecycleGeneration">本次选择所属的局生命周期。</param>
    /// <param name="ActionSource">触发选择的战斗动作来源。</param>
    private sealed record CombatPileSelectionCapture(
        CardModel[] Options,
        CardSelectorPrefs Prefs,
        long LifecycleGeneration,
        string ActionSource)
    {
        /// <summary>保存可见网格的真实候选顺序。</summary>
        public CardModel[]? VisibleOptions { get; set; }

        /// <summary>可见点击调用栈能够提供时保存更直接的动作来源。</summary>
        public string? VisibleActionSource { get; set; }

        /// <summary>优先使用可见点击来源，否则继承触发本次选择的战斗动作。</summary>
        public string EffectiveActionSource =>
            VisibleActionSource ?? ActionSource;

        /// <summary>按卡牌实例保存点击时构造的真实动作前状态。</summary>
        public Dictionary<CardModel, Capture> VisibleCaptures { get; } =
            new(ReferenceEqualityComparer.Instance);

        /// <summary>保存最终仍被选中的可见点击顺序。</summary>
        public List<CardModel> VisibleSelectionOrder { get; } = new();
    }

    /// <summary>保存一次无需显示界面的原版手牌选择。</summary>
    /// <param name="Options">原版过滤后的手牌候选。</param>
    /// <param name="Prefs">原版选择数量与确认规则。</param>
    /// <param name="LifecycleGeneration">本次选择所属的局生命周期。</param>
    /// <param name="ActionSource">触发选择的战斗动作来源。</param>
    private sealed record ImplicitHandSelectionCapture(
        CardModel[] Options,
        CardSelectorPrefs Prefs,
        long LifecycleGeneration,
        string ActionSource);

    private static NativeAttempt? BeginGridConfirmation(
        NCardGridSelectionScreen screen)
    {
        return GameStateService.TryGetGridSelectionMetadata(
                   screen, out _, out var selection) &&
               selection.MaxSelect > 1
            ? Attempt("confirm_selection")
            : null;
    }

    public static void Start()
    {
        if (_harmony != null) return;
        _harmony = new Harmony("sts2-ai-agent.native-ui-actions");
        _harmony.PatchAll(typeof(NativeUiActionRecorder).Assembly);
    }

    public static IDisposable Suppress()
    {
        Interlocked.Increment(ref _suppressionDepth);
        return new SuppressionScope();
    }

    private sealed class SuppressionScope : IDisposable
    {
        private int _disposed;
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
                Interlocked.Decrement(ref _suppressionDepth);
        }
    }

    private static Capture? Begin(string action, int? cardIndex = null,
        int? optionIndex = null, int? targetIndex = null,
        bool trustNativeUiGate = false,
        string? actionSource = null)
    {
        if (Volatile.Read(ref _suppressionDepth) != 0) return null;
        try
        {
            actionSource ??= ResolveActionSource();
            var generation = GameEventService.Instance.CaptureGeneration();
            var before = GameStateService.BuildStatePayload(
                trustNativeUiGate ? action : null);
            if (!before.available_actions.Contains(action, StringComparer.Ordinal))
                return null;
            var rewardAction = action is "choose_reward_card" or
                "skip_reward_cards" or "choose_reward_alternative";
            var layer = (before.screen == "COMBAT" || before.in_combat) &&
                !rewardAction ? "battle" : "strategic";
            return new Capture(new ActionRequest
            {
                action = action,
                card_index = cardIndex,
                option_index = optionIndex,
                target_index = targetIndex,
                client_context = new { source = actionSource, layer }
            }, before, generation, actionSource);
        }
        catch { return null; }
    }

    /// <summary>判断当前原生游戏调用是否由 CombatSolver 发起。</summary>
    /// <returns>调用栈中包含 CombatSolver 程序集时返回 <see langword="true"/>。</returns>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool IsCombatSolverCall()
    {
        return new StackTrace().GetFrames().Any(frame =>
            string.Equals(
                frame.GetMethod()?.DeclaringType?.Assembly.GetName().Name,
                CombatSolverAssemblyName,
                StringComparison.Ordinal));
    }

    /// <summary>从当前同步调用栈解析动作的真实执行者。</summary>
    /// <returns>CombatSolver 自动执行或人类原生 UI 的来源标记。</returns>
    private static string ResolveActionSource()
    {
        return IsCombatSolverCall() ? CombatSolverSource : HumanUiSource;
    }

    private static void Finish(Capture? capture)
    {
        if (capture == null) return;
        try
        {
            GameEventService.Instance.PublishActionExecuted(
                capture.Request, capture.Before, new ActionResponsePayload
                {
                    action = capture.Request.action ?? string.Empty,
                    status = "accepted",
                    stable = false,
                    message = "Native UI action accepted.",
                    state = GameStateService.BuildStatePayload()
                }, capture.LifecycleGeneration);
        }
        catch
        {
            CaptureGap(capture.Request.action ?? "unknown",
                "native action publication failed",
                capture.LifecycleGeneration,
                capture.ActionSource);
        }
    }

    private static NativeAttempt? Attempt(string action, int? cardIndex = null,
        int? optionIndex = null, int? targetIndex = null)
    {
        if (Volatile.Read(ref _suppressionDepth) != 0) return null;
        var capture = Begin(action, cardIndex, optionIndex, targetIndex);
        return capture == null ? null : new NativeAttempt(action, capture);
    }

    private static void FinishOrGap(string action, Capture? capture)
    {
        if (Volatile.Read(ref _suppressionDepth) != 0) return;
        if (capture == null)
        {
            CaptureGap(action,
                "native action committed without before-state capture");
            return;
        }
        Finish(capture);
    }

    private static void FinishAttempt(NativeAttempt? attempt)
    {
        if (attempt != null) FinishOrGap(attempt.Action, attempt.Capture);
    }

    private static int RefIndex<T>(IEnumerable<T> values, T target) where T : class
    {
        var index = 0;
        foreach (var value in values)
        {
            if (ReferenceEquals(value, target)) return index;
            index++;
        }
        return -1;
    }

    private static int? TargetIndex(CombatState? combat, Creature? target)
    {
        if (combat == null || target == null) return null;
        var index = RefIndex(combat.Enemies, target);
        if (index >= 0) return index;
        var players = combat.Players
            .OrderBy(p => combat.RunState is RunState run
                ? run.GetPlayerSlotIndex(p) : 0)
            .Select(p => p.Creature);
        index = RefIndex(players, target);
        return index >= 0 ? index : null;
    }

    private static Capture WithTarget(Capture capture, int? targetIndex)
    {
        var request = capture.Request;
        return new Capture(new ActionRequest
        {
            action = request.action,
            card_index = request.card_index,
            option_index = request.option_index,
            target_index = targetIndex,
            command = request.command,
            client_context = request.client_context
        }, capture.Before, capture.LifecycleGeneration, capture.ActionSource);
    }

    private static void CaptureGap(
        string action,
        string reason,
        long? generation = null,
        string? actionSource = null)
    {
        if (Volatile.Read(ref _suppressionDepth) != 0) return;
        try
        {
            generation ??= GameEventService.Instance.CaptureGeneration();
            actionSource ??= ResolveActionSource();
            GameEventService.Instance.PublishNativeUiCaptureGap(
                action, reason, actionSource, generation.Value);
        }
        catch { }
    }

    private static Capture? BeginCard(CardModel card, Creature? target,
        bool trustNativeUiGate = false,
        string? actionSource = null)
    {
        var combat = CombatManager.Instance.DebugOnlyGetState();
        var cards = GameStateService.GetLocalPlayer(combat)?
            .PlayerCombatState?.Hand.Cards;
        var index = cards == null ? -1 : RefIndex(cards, card);
        return index < 0 ? null : Begin("play_card", cardIndex: index,
            targetIndex: TargetIndex(combat, target),
            trustNativeUiGate: trustNativeUiGate,
            actionSource: actionSource);
    }

    private static Capture? BeginPotion(PotionModel potion, Creature? target,
        bool trustNativeUiGate = false,
        string? actionSource = null)
    {
        var combat = CombatManager.Instance.DebugOnlyGetState();
        var match = potion.Owner.PotionSlots
            .Select((candidate, index) => new { candidate, index })
            .FirstOrDefault(x => ReferenceEquals(x.candidate, potion));
        var index = match?.index ?? -1;
        return index < 0 ? null : Begin("use_potion", optionIndex: index,
            targetIndex: GameStateService.PotionRequiresTarget(combat, potion)
                ? TargetIndex(combat, target) : null,
            trustNativeUiGate: trustNativeUiGate,
            actionSource: actionSource);
    }

    private static Capture? BeginSelectedCard(CardModel card)
    {
        return BeginSelectedCard(
            ActiveScreenContext.Instance.GetCurrentScreen(),
            card);
    }

    /// <summary>使用已经确定的界面查找选牌索引，避免再次读取滞后的活动界面。</summary>
    /// <param name="screen">实际收到点击的选牌界面。</param>
    /// <param name="card">玩家或 Solver 选择的卡牌。</param>
    /// <returns>动作前捕获；界面中找不到卡牌时为 <see langword="null"/>。</returns>
    private static Capture? BeginSelectedCard(
        IScreenContext? screen,
        CardModel card)
    {
        var holders = GameStateService.GetDeckSelectionOptions(screen);
        var match = holders.Select((holder, index) => new { holder, index })
            .FirstOrDefault(x => ReferenceEquals(x.holder.CardModel, card));
        return match == null ? null
            : Begin("select_deck_card", optionIndex: match.index);
    }

    /// <summary>保存最近一次实际提交的战斗动作来源，供其异步选牌效果继承。</summary>
    /// <param name="source">提交卡牌或药水的执行者。</param>
    private static void RememberCombatActionSource(string source)
    {
        _lastCombatActionSource = source;
        _lastCombatActionGeneration =
            GameEventService.Instance.CaptureGeneration();
    }

    /// <summary>从真实候选和已选卡牌构造一条牌堆选择动作。</summary>
    /// <param name="selection">当前原版牌堆选择。</param>
    /// <param name="options">玩家可见的候选顺序。</param>
    /// <param name="selectedCards">本条动作前已经选中的牌。</param>
    /// <param name="card">本次实际选择的牌。</param>
    /// <returns>可发布的动作捕获；目标卡牌不在候选中时为 <see langword="null"/>。</returns>
    private static Capture? BuildCombatPileCardCapture(
        CombatPileSelectionCapture selection,
        IReadOnlyList<CardModel> options,
        IReadOnlySet<CardModel> selectedCards,
        CardModel card)
    {
        var optionIndex = RefIndex(options, card);
        if (optionIndex < 0) return null;
        try
        {
            var before = GameStateService.BuildCombatPileSelectionState(
                options,
                selection.Prefs,
                selectedCards);
            return new Capture(new ActionRequest
            {
                action = "select_deck_card",
                option_index = optionIndex,
                client_context = new
                {
                    source = selection.EffectiveActionSource,
                    layer = "battle"
                }
            }, before, selection.LifecycleGeneration,
                selection.EffectiveActionSource);
        }
        catch { return null; }
    }

    /// <summary>等待原版牌堆选择完成，并按真实返回结果发布选择动作。</summary>
    /// <param name="task">原版选牌任务。</param>
    /// <param name="selection">调用前保存的候选与来源。</param>
    /// <returns>不改变内容和顺序的原版选择结果。</returns>
    private static async Task<IEnumerable<CardModel>> FinishCombatPileSelectionAsync(
        Task<IEnumerable<CardModel>> task,
        CombatPileSelectionCapture selection)
    {
        try
        {
            var result = (await task).ToArray();
            var selectedCards = new HashSet<CardModel>(
                ReferenceEqualityComparer.Instance);
            var options = selection.VisibleOptions ?? selection.Options;
            var orderedResult = selection.VisibleSelectionOrder
                .Where(card => result.Any(candidate =>
                    ReferenceEquals(candidate, card)))
                .Concat(result.Where(card =>
                    !selection.VisibleSelectionOrder.Any(candidate =>
                        ReferenceEquals(candidate, card))))
                .ToArray();
            foreach (var card in orderedResult)
            {
                var capture = selection.VisibleCaptures.GetValueOrDefault(card)
                    ?? BuildCombatPileCardCapture(
                        selection,
                        options,
                        selectedCards,
                        card);
                if (capture == null)
                {
                    CaptureGap(
                        "select_deck_card",
                        "combat pile returned a card without a complete before-state",
                        selection.LifecycleGeneration,
                        selection.EffectiveActionSource);
                }
                else
                {
                    Finish(capture);
                }
                selectedCards.Add(card);
            }

            if (result.Length > 0 && selection.Prefs.RequireManualConfirmation)
            {
                try
                {
                    var before = GameStateService.BuildCombatPileSelectionState(
                        options,
                        selection.Prefs,
                        selectedCards);
                    Finish(new Capture(new ActionRequest
                    {
                        action = "confirm_selection",
                        client_context = new
                        {
                            source = selection.EffectiveActionSource,
                            layer = "battle"
                        }
                    }, before, selection.LifecycleGeneration,
                        selection.EffectiveActionSource));
                }
                catch
                {
                    CaptureGap(
                        "confirm_selection",
                        "combat pile confirmation lacks a complete before-state",
                        selection.LifecycleGeneration,
                        selection.EffectiveActionSource);
                }
            }
            return result;
        }
        finally
        {
            if (ReferenceEquals(_activeCombatPileSelection, selection))
                _activeCombatPileSelection = null;
        }
    }

    /// <summary>等待隐式手牌选择完成，并按原版真实返回结果发布动作。</summary>
    /// <param name="task">原版手牌选择任务。</param>
    /// <param name="selection">调用前保存的候选与来源。</param>
    /// <returns>不改变内容和顺序的原版选择结果。</returns>
    private static async Task<IEnumerable<CardModel>> FinishImplicitHandSelectionAsync(
        Task<IEnumerable<CardModel>> task,
        ImplicitHandSelectionCapture selection)
    {
        var result = (await task).ToArray();
        var selectedCards = new HashSet<CardModel>(
            ReferenceEqualityComparer.Instance);
        foreach (var card in result)
        {
            var optionIndex = RefIndex(selection.Options, card);
            Capture? capture = null;
            if (optionIndex >= 0)
            {
                try
                {
                    var before = GameStateService.BuildCombatHandSelectionState(
                        selection.Options,
                        selection.Prefs,
                        selectedCards);
                    capture = new Capture(new ActionRequest
                    {
                        action = "select_deck_card",
                        option_index = optionIndex,
                        client_context = new
                        {
                            source = selection.ActionSource,
                            layer = "battle"
                        }
                    }, before, selection.LifecycleGeneration,
                        selection.ActionSource);
                }
                catch { }
            }

            if (capture == null)
            {
                CaptureGap(
                    "select_deck_card",
                    "implicit hand selection returned a card without a complete before-state",
                    selection.LifecycleGeneration,
                    selection.ActionSource);
            }
            else
            {
                Finish(capture);
            }
            selectedCards.Add(card);
        }
        return result;
    }

    private static IReadOnlyList<CardModel>? GetSelectedGridCards(
        NCardGridSelectionScreen screen)
    {
        try
        {
            var field = AccessTools.Field(screen.GetType(), "_selectedCards");
            return field?.GetValue(screen) is IEnumerable<CardModel> cards
                ? cards.ToArray()
                : null;
        }
        catch { return null; }
    }

    private static void ClearGridSelection(
        NCardGridSelectionScreen? screen = null)
    {
        if (screen != null && !ReferenceEquals(screen, _pendingGridScreen))
            return;
        _pendingGridSelections.Clear();
        _pendingGridScreen = null;
    }

    private static void QueueGridSelection(
        NCardGridSelectionScreen screen, CardModel card)
    {
        if (Volatile.Read(ref _suppressionDepth) != 0) return;
        if (!ReferenceEquals(screen, _pendingGridScreen))
        {
            ClearGridSelection();
            _pendingGridScreen = screen;
        }

        var selected = GetSelectedGridCards(screen);
        if (selected == null)
        {
            CaptureGap("select_deck_card",
                "grid selection state unavailable at native click");
            ClearGridSelection(screen);
            return;
        }

        if (screen is NCombatPileCardSelectScreen &&
            _activeCombatPileSelection is { } combatPileSelection)
        {
            combatPileSelection.VisibleActionSource = ResolveActionSource();
            var options = GameStateService.GetDeckSelectionOptions(screen)
                .Select(holder => holder.CardModel)
                .Where(candidate => candidate != null)
                .Cast<CardModel>()
                .ToArray();
            combatPileSelection.VisibleOptions = options;
            if (selected.Any(candidate => ReferenceEquals(candidate, card)))
            {
                combatPileSelection.VisibleCaptures.Remove(card);
                combatPileSelection.VisibleSelectionOrder.RemoveAll(candidate =>
                    ReferenceEquals(candidate, card));
                return;
            }
            var combatCapture = BuildCombatPileCardCapture(
                combatPileSelection,
                options,
                new HashSet<CardModel>(
                    selected,
                    ReferenceEqualityComparer.Instance),
                card);
            if (combatCapture != null)
            {
                combatPileSelection.VisibleCaptures[card] = combatCapture;
                combatPileSelection.VisibleSelectionOrder.RemoveAll(candidate =>
                    ReferenceEquals(candidate, card));
                combatPileSelection.VisibleSelectionOrder.Add(card);
            }
            return;
        }

        var pendingIndex = _pendingGridSelections.FindIndex(
            pending => ReferenceEquals(pending.Card, card));
        if (selected.Any(candidate => ReferenceEquals(candidate, card)))
        {
            if (pendingIndex >= 0) _pendingGridSelections.RemoveAt(pendingIndex);
            return;
        }

        var capture = BeginSelectedCard(screen, card);
        if (capture == null) return;
        if (pendingIndex >= 0) _pendingGridSelections.RemoveAt(pendingIndex);
        _pendingGridSelections.Add((card, capture));
    }

    private static Capture[] TakeCommittedGridSelections(
        NCardGridSelectionScreen screen)
    {
        if (Volatile.Read(ref _suppressionDepth) != 0)
        {
            ClearGridSelection(screen);
            return Array.Empty<Capture>();
        }

        var selected = GetSelectedGridCards(screen);
        var captures = selected == null
            ? Array.Empty<Capture>()
            : _pendingGridSelections
                .Where(pending => selected.Any(card =>
                    ReferenceEquals(card, pending.Card)))
                .Select(pending => pending.Capture)
                .ToArray();
        if (selected == null || captures.Length != selected.Count)
        {
            CaptureGap("select_deck_card",
                "grid selection committed without matching native clicks");
            captures = Array.Empty<Capture>();
        }
        ClearGridSelection(screen);
        return captures;
    }

    private static NativeAttempt? MapButton(NButton button)
    {
        var current = ActiveScreenContext.Instance.GetCurrentScreen();
        if (ReferenceEquals(button, GameStateService.GetModalConfirmButton(current)))
            return Attempt("confirm_modal");
        if (ReferenceEquals(button, GameStateService.GetModalCancelButton(current)))
            return Attempt("dismiss_modal");

        if (button is NMapPoint mapPoint)
        {
            var index = RefIndex(GameStateService.GetAvailableMapNodes(
                current, RunManager.Instance.DebugOnlyGetState()), mapPoint);
            return new NativeAttempt("choose_map_node", index < 0 ? null
                : Begin("choose_map_node", optionIndex: index));
        }
        if (button is NProceedButton)
        {
            return Attempt("proceed");
        }
        if (button is NMerchantButton) return Attempt("open_shop_inventory");
        if (button is NTreasureRoomRelicHolder relic)
            return Attempt("choose_treasure_relic", optionIndex: relic.Index);

        if (GameStateService.GetBundleConfirmButtons(current)
            .Any(candidate => ReferenceEquals(candidate, button)))
            return Attempt("confirm_bundle");
        return null;
    }

    private static Capture? BeginReward(Reward reward)
    {
        var buttons = GameStateService.GetRewardButtons(
            ActiveScreenContext.Instance.GetCurrentScreen());
        var match = buttons.Select((button, index) => new { button, index })
            .FirstOrDefault(x => ReferenceEquals(x.button.Reward, reward));
        return match == null ? null : Begin("claim_reward", optionIndex: match.index);
    }

    private static Capture? BeginRewardButton(NRewardButton button)
    {
        var buttons = GameStateService.GetRewardButtons(
            ActiveScreenContext.Instance.GetCurrentScreen());
        var index = RefIndex(buttons, button);
        return index < 0 ? null : Begin("claim_reward", optionIndex: index,
            trustNativeUiGate: true);
    }

    private static Capture? TakePendingRewardCapture(Reward reward)
    {
        var capture = ReferenceEquals(_pendingReward, reward)
            ? _pendingRewardCapture : null;
        _pendingReward = null;
        _pendingRewardCapture = null;
        return capture;
    }

    private static Capture? BeginCrystalSphereCell(NCrystalSphereCell cell)
    {
        var index = RefIndex(GameStateService.GetClickableCrystalSphereCells(
            ActiveScreenContext.Instance.GetCurrentScreen()), cell);
        return index < 0 ? null
            : Begin("choose_crystal_sphere_cell", optionIndex: index);
    }

    private static NativeAttempt? BeginMerchant(MerchantEntry entry)
    {
        var current = ActiveScreenContext.Instance.GetCurrentScreen();
        if (entry is MerchantCardEntry card)
        {
            var index = RefIndex(GameStateService.GetMerchantCardEntries(current), card);
            return new NativeAttempt("buy_card", index < 0 ? null
                : Begin("buy_card", optionIndex: index));
        }
        if (entry is MerchantRelicEntry relic)
        {
            var index = RefIndex(GameStateService.GetMerchantRelicEntries(current), relic);
            return new NativeAttempt("buy_relic", index < 0 ? null
                : Begin("buy_relic", optionIndex: index));
        }
        if (entry is MerchantPotionEntry potion)
        {
            var index = RefIndex(GameStateService.GetMerchantPotionEntries(current), potion);
            return new NativeAttempt("buy_potion", index < 0 ? null
                : Begin("buy_potion", optionIndex: index));
        }
        return null;
    }

    private static async Task<bool> FinishMerchantIfTrue(
        Task<bool> task, NativeAttempt attempt)
    {
        var result = await task;
        if (result) FinishAttempt(attempt);
        return result;
    }

    private static async Task<bool> FinishRewardIfTrue(
        Task<bool> task, NativeAttempt attempt)
    {
        var result = await task;
        if (result) FinishAttempt(attempt);
        return result;
    }

    [HarmonyPatch(typeof(NTreasureRoom), "OnChestButtonReleased")]
    private static class ChestOpenPatch
    {
        static void Prefix(out Capture? __state) =>
            __state = Begin("open_chest");
        static void Postfix(Capture? __state) =>
            FinishOrGap("open_chest", __state);
    }

    [HarmonyPatch(typeof(NMerchantInventory), "Close")]
    private static class ShopInventoryClosePatch
    {
        static void Prefix(out Capture? __state) =>
            __state = Begin("close_shop_inventory");
        static void Postfix(Capture? __state) =>
            FinishOrGap("close_shop_inventory", __state);
    }

    [HarmonyPatch(typeof(NPlayerHand), "StartCardPlay")]
    private static class CardPlayStartPatch
    {
        static void Prefix(NHandCardHolder holder)
        {
            _pendingCard = holder.CardModel;
            _pendingCardCapture = holder.CardModel is CardModel card
                ? BeginCard(card, null, trustNativeUiGate: true) : null;
        }
    }

    [HarmonyPatch(typeof(CardModel), nameof(CardModel.TryManualPlay))]
    private static class CardPlayPatch
    {
        static void Prefix(CardModel __instance, Creature? target,
            out CommitCapture? __state)
        {
            __state = null;
            if (Volatile.Read(ref _suppressionDepth) != 0) return;
            var capture = _pendingCardCapture;
            var matches = ReferenceEquals(_pendingCard, __instance);
            _pendingCardCapture = null;
            _pendingCard = null;
            if (matches)
            {
                var source = capture?.ActionSource ?? HumanUiSource;
                __state = new CommitCapture(capture == null ? null : WithTarget(
                    capture, TargetIndex(
                        CombatManager.Instance.DebugOnlyGetState(), target)), source);
                return;
            }
            if (!IsCombatSolverCall()) return;
            __state = new CommitCapture(BeginCard(
                __instance,
                target,
                trustNativeUiGate: true,
                actionSource: CombatSolverSource), CombatSolverSource);
        }
        static void Postfix(CardModel __instance, bool __result,
            CommitCapture? __state)
        {
            if (!__result) return;
            if (Volatile.Read(ref _suppressionDepth) != 0) return;
            RememberCombatActionSource(
                __state?.ActionSource ?? HumanUiSource);
            GameActionService.RecordCardPlayed(
                CombatManager.Instance.DebugOnlyGetState()?.RoundNumber ?? 0,
                __instance.Type.ToString());
            if (__state?.Capture != null)
                Finish(__state.Capture);
            else if (__state != null || _cardUiCommitDepth > 0)
                CaptureGap("play_card",
                    "native card commit succeeded without start capture",
                    actionSource: __state?.ActionSource ?? HumanUiSource);
        }
    }

    [HarmonyPatch(typeof(NCardPlay), "TryPlayCard")]
    private static class NativeCardCommitPatch
    {
        static void Prefix() => _cardUiCommitDepth++;
        static void Postfix() => _cardUiCommitDepth--;
    }

    [HarmonyPatch(typeof(NPotionHolder), nameof(NPotionHolder.UsePotion))]
    private static class PotionUseStartPatch
    {
        static void Prefix(NPotionHolder __instance)
        {
            _pendingPotion = __instance.Potion?.Model;
            _pendingPotionCapture = _pendingPotion == null
                ? null : BeginPotion(
                    _pendingPotion, null, trustNativeUiGate: true);
        }
    }

    [HarmonyPatch(typeof(PotionModel), nameof(PotionModel.EnqueueManualUse))]
    private static class PotionUsePatch
    {
        static void Prefix(PotionModel __instance, Creature? target,
            out CommitCapture? __state)
        {
            __state = null;
            if (Volatile.Read(ref _suppressionDepth) != 0) return;
            var capture = _pendingPotionCapture;
            var matches = ReferenceEquals(_pendingPotion, __instance);
            _pendingPotionCapture = null;
            _pendingPotion = null;
            if (matches)
            {
                var source = capture?.ActionSource ?? HumanUiSource;
                var combat = CombatManager.Instance.DebugOnlyGetState();
                __state = new CommitCapture(capture == null ? null : WithTarget(
                    capture,
                    GameStateService.PotionRequiresTarget(combat, __instance)
                        ? TargetIndex(combat, target) : null), source);
                return;
            }
            if (!IsCombatSolverCall()) return;
            __state = new CommitCapture(BeginPotion(
                __instance,
                target,
                trustNativeUiGate: true,
                actionSource: CombatSolverSource), CombatSolverSource);
        }
        static void Postfix(CommitCapture? __state)
        {
            if (__state != null)
                RememberCombatActionSource(__state.ActionSource);
            if (__state?.Capture != null)
            {
                Finish(__state.Capture);
                return;
            }
            if (__state != null)
                CaptureGap("use_potion",
                    "native potion commit succeeded without start capture",
                    actionSource: __state.ActionSource);
        }
    }

    [HarmonyPatch(typeof(DiscardPotionGameAction), MethodType.Constructor,
        typeof(Player), typeof(uint), typeof(bool))]
    private static class PotionDiscardPatch
    {
        static void Prefix(Player player, uint potionSlotIndex,
            out NativeAttempt? __state)
        {
            var local = GameStateService.GetLocalPlayer(
                RunManager.Instance.DebugOnlyGetState());
            __state = ReferenceEquals(player, local)
                ? Attempt("discard_potion", optionIndex: (int)potionSlotIndex)
                : null;
        }
        static void Postfix(NativeAttempt? __state) => FinishAttempt(__state);
    }

    [HarmonyPatch(typeof(NEndTurnButton), nameof(NEndTurnButton.CallReleaseLogic))]
    private static class EndTurnPatch
    {
        static void Prefix(out Capture? __state) => __state = Begin("end_turn");
        static void Postfix(Capture? __state) => Finish(__state);
    }

    /// <summary>观察 CombatSolver 实际入队的结束回合动作。</summary>
    [HarmonyPatch(typeof(EndPlayerTurnAction), MethodType.Constructor,
        typeof(Player), typeof(int))]
    private static class CombatSolverEndTurnPatch
    {
        /// <summary>在 Solver 构造本地玩家的结束回合动作前保存状态。</summary>
        /// <param name="player">准备结束回合的玩家。</param>
        /// <param name="turnNumber">动作绑定的战斗回合编号。</param>
        /// <param name="__state">返回给后置补丁的动作捕获。</param>
        static void Prefix(Player player, int turnNumber,
            out CommitCapture? __state)
        {
            _ = turnNumber;
            var local = GameStateService.GetLocalPlayer(
                CombatManager.Instance.DebugOnlyGetState());
            __state = !IsCombatSolverCall() || !ReferenceEquals(player, local)
                ? null
                : new CommitCapture(
                    Begin(
                        "end_turn",
                        trustNativeUiGate: true,
                        actionSource: CombatSolverSource),
                    CombatSolverSource);
        }

        /// <summary>在 Solver 结束回合动作构造后发布动作或缺口。</summary>
        /// <param name="__state">前置补丁保存的动作捕获。</param>
        static void Postfix(CommitCapture? __state)
        {
            if (__state?.Capture != null)
            {
                Finish(__state.Capture);
                return;
            }
            if (__state != null)
                CaptureGap("end_turn",
                    "solver end-turn commit succeeded without before-state capture",
                    actionSource: __state.ActionSource);
        }
    }

    [HarmonyPatch]
    private static class ButtonPatch
    {
        static IEnumerable<MethodBase> TargetMethods() => typeof(NButton).Assembly
            .GetTypes().Where(t => typeof(NButton).IsAssignableFrom(t))
            .Select(t => AccessTools.DeclaredMethod(t, "OnRelease"))
            .Where(m => m != null).Distinct()!;
        static void Prefix(NButton __instance, out NativeAttempt? __state)
        {
            __state = _buttonPatchDepth++ == 0 ? MapButton(__instance) : null;
        }
        static void Postfix(NativeAttempt? __state)
        {
            _buttonPatchDepth--;
            FinishAttempt(__state);
        }
    }

    [HarmonyPatch(typeof(NCardRewardSelectionScreen), "SelectCard")]
    private static class RewardCardPatch
    {
        static void Prefix(NCardHolder cardHolder, out Capture? __state)
        {
            var index = RefIndex(GameStateService.GetCardRewardOptions(
                ActiveScreenContext.Instance.GetCurrentScreen()), cardHolder);
            __state = index < 0 ? null
                : Begin("choose_reward_card", optionIndex: index);
        }
        static void Postfix(Capture? __state) =>
            FinishOrGap("choose_reward_card", __state);
    }

    [HarmonyPatch]
    private static class GridCardPatch
    {
        static IEnumerable<MethodBase> TargetMethods() =>
            typeof(NCardGridSelectionScreen).Assembly.GetTypes()
                .Where(t => !t.IsAbstract &&
                    typeof(NCardGridSelectionScreen).IsAssignableFrom(t))
                .Select(t => AccessTools.DeclaredMethod(t, "OnCardClicked"))
                .Where(m => m != null).Distinct()!;
        static void Prefix(NCardGridSelectionScreen __instance, CardModel card) =>
            QueueGridSelection(__instance, card);
    }

    [HarmonyPatch]
    private static class GridSelectionCommitPatch
    {
        static IEnumerable<MethodBase> TargetMethods()
        {
            var methods = new MethodBase?[]
            {
                AccessTools.DeclaredMethod(typeof(NDeckCardSelectScreen),
                    "ConfirmSelection"),
                AccessTools.DeclaredMethod(typeof(NDeckUpgradeSelectScreen),
                    "ConfirmSelection"),
                AccessTools.DeclaredMethod(typeof(NDeckEnchantSelectScreen),
                    "ConfirmSelection"),
                AccessTools.DeclaredMethod(typeof(NDeckTransformSelectScreen),
                    "CompleteSelection"),
                AccessTools.DeclaredMethod(typeof(NSimpleCardSelectScreen),
                    "CompleteSelection"),
                AccessTools.DeclaredMethod(typeof(NCombatPileCardSelectScreen),
                    "CompleteSelection")
            };
            return methods.Where(method => method != null)!;
        }
        static void Prefix(NCardGridSelectionScreen __instance,
            out GridCommitCapture __state)
        {
            __state = __instance is NCombatPileCardSelectScreen
                ? new GridCommitCapture(Array.Empty<Capture>(), null)
                : new GridCommitCapture(
                    TakeCommittedGridSelections(__instance),
                    BeginGridConfirmation(__instance));
        }
        static void Postfix(GridCommitCapture __state)
        {
            foreach (var capture in __state.Selections) Finish(capture);
            FinishAttempt(__state.Confirmation);
        }
    }

    /// <summary>在原版返回真实牌堆选择结果后统一发布可见和隐式动作。</summary>
    [HarmonyPatch(typeof(CardSelectCmd), nameof(CardSelectCmd.FromCombatPile),
        typeof(PlayerChoiceContext), typeof(CardPile), typeof(Player),
        typeof(CardSelectorPrefs), typeof(Func<CardModel, bool>))]
    private static class CombatPileSelectionPatch
    {
        /// <summary>保存过滤后的候选、动作来源和局生命周期。</summary>
        /// <param name="pile">原版准备展示或隐式选择的战斗牌堆。</param>
        /// <param name="player">执行选择的玩家。</param>
        /// <param name="prefs">原版选择数量和确认规则。</param>
        /// <param name="filter">原版用于限制候选牌的过滤器。</param>
        /// <param name="__state">返回给后置补丁的选牌上下文。</param>
        static void Prefix(
            CardPile pile,
            Player player,
            CardSelectorPrefs prefs,
            Func<CardModel, bool>? filter,
            out CombatPileSelectionCapture? __state)
        {
            __state = null;
            if (Volatile.Read(ref _suppressionDepth) != 0) return;
            var local = GameStateService.GetLocalPlayer(
                CombatManager.Instance.DebugOnlyGetState());
            if (!ReferenceEquals(player, local)) return;
            var options = (filter == null
                    ? pile.Cards
                    : pile.Cards.Where(filter))
                .ToArray();
            if (options.Length == 0) return;
            var generation = GameEventService.Instance.CaptureGeneration();
            __state = new CombatPileSelectionCapture(
                options,
                prefs,
                generation,
                _lastCombatActionGeneration == generation
                    ? _lastCombatActionSource
                    : ResolveActionSource());
            _activeCombatPileSelection = __state;
        }

        /// <summary>包装异步返回值，在调用方应用选择结果前发布动作。</summary>
        /// <param name="__result">原版异步选择结果。</param>
        /// <param name="__state">前置补丁保存的选牌上下文。</param>
        static void Postfix(
            ref Task<IEnumerable<CardModel>> __result,
            CombatPileSelectionCapture? __state)
        {
            if (__state != null)
                __result = FinishCombatPileSelectionAsync(__result, __state);
        }
    }

    /// <summary>捕获原版不会显示手牌界面的单候选隐式选择。</summary>
    [HarmonyPatch(typeof(CardSelectCmd), nameof(CardSelectCmd.FromHand),
        typeof(PlayerChoiceContext), typeof(Player), typeof(CardSelectorPrefs),
        typeof(Func<CardModel, bool>), typeof(AbstractModel))]
    private static class ImplicitHandSelectionPatch
    {
        /// <summary>在原版直接返回候选前保存候选、来源和局生命周期。</summary>
        /// <param name="player">被选择手牌的玩家。</param>
        /// <param name="prefs">原版选择数量和确认规则。</param>
        /// <param name="filter">原版用于限制候选牌的过滤器。</param>
        /// <param name="__state">返回给后置补丁的隐式选择上下文。</param>
        static void Prefix(
            Player player,
            CardSelectorPrefs prefs,
            Func<CardModel, bool>? filter,
            out ImplicitHandSelectionCapture? __state)
        {
            __state = null;
            if (Volatile.Read(ref _suppressionDepth) != 0) return;
            var local = GameStateService.GetLocalPlayer(
                CombatManager.Instance.DebugOnlyGetState());
            if (!ReferenceEquals(player, local)) return;
            var options = PileType.Hand.GetPile(player).Cards
                .Where(filter ?? (_ => true))
                .ToArray();
            if (options.Length == 0 || prefs.RequireManualConfirmation ||
                options.Length > prefs.MinSelect)
            {
                return;
            }

            var generation = GameEventService.Instance.CaptureGeneration();
            __state = new ImplicitHandSelectionCapture(
                options,
                prefs,
                generation,
                _lastCombatActionGeneration == generation
                    ? _lastCombatActionSource
                    : ResolveActionSource());
        }

        /// <summary>包装异步返回值，在调用方移动所选手牌前发布动作。</summary>
        /// <param name="__result">原版异步选择结果。</param>
        /// <param name="__state">前置补丁保存的隐式选择上下文。</param>
        static void Postfix(
            ref Task<IEnumerable<CardModel>> __result,
            ImplicitHandSelectionCapture? __state)
        {
            if (__state != null)
            {
                __result = FinishImplicitHandSelectionAsync(__result, __state);
            }
        }
    }

    [HarmonyPatch]
    private static class GridSelectionCancelPatch
    {
        static IEnumerable<MethodBase> TargetMethods() =>
            typeof(NCardGridSelectionScreen).Assembly.GetTypes()
                .Where(t => !t.IsAbstract &&
                    typeof(NCardGridSelectionScreen).IsAssignableFrom(t))
                .Select(t => AccessTools.DeclaredMethod(t, "CancelSelection"))
                .Where(method => method != null).Distinct()!;
        static void Prefix(NCardGridSelectionScreen __instance) =>
            ClearGridSelection(__instance);
    }

    [HarmonyPatch(typeof(NCardGridSelectionScreen), "_ExitTree")]
    private static class GridSelectionExitPatch
    {
        static void Prefix(NCardGridSelectionScreen __instance) =>
            ClearGridSelection(__instance);
    }

    [HarmonyPatch(typeof(NChooseACardSelectionScreen), "SelectHolder")]
    private static class ChooseCardPatch
    {
        static void Prefix(NCardHolder cardHolder, out Capture? __state)
        {
            var index = RefIndex(GameStateService.GetDeckSelectionOptions(
                ActiveScreenContext.Instance.GetCurrentScreen()), cardHolder);
            __state = index < 0 ? null
                : Begin("select_deck_card", optionIndex: index);
        }
        static void Postfix(Capture? __state) =>
            FinishOrGap("select_deck_card", __state);
    }

    [HarmonyPatch(typeof(NChooseACardSelectionScreen), "OnSkipButtonReleased")]
    private static class ChooseCardSkipPatch
    {
        static void Prefix(out Capture? __state) =>
            __state = Begin("skip_card_selection");
        static void Postfix(Capture? __state) =>
            FinishOrGap("skip_card_selection", __state);
    }

    [HarmonyPatch]
    private static class HandCardPatch
    {
        static IEnumerable<MethodBase> TargetMethods()
        {
            yield return AccessTools.DeclaredMethod(typeof(NPlayerHand),
                "SelectCardInSimpleMode");
            yield return AccessTools.DeclaredMethod(typeof(NPlayerHand),
                "SelectCardInUpgradeMode");
        }
        static void Prefix(NHandCardHolder holder, out Capture? __state) =>
            __state = holder.CardModel is CardModel card
                ? BeginSelectedCard(card) : null;
        static void Postfix(Capture? __state) =>
            FinishOrGap("select_deck_card", __state);
    }

    [HarmonyPatch(typeof(NPlayerHand), "OnSelectModeConfirmButtonPressed")]
    private static class HandConfirmPatch
    {
        static void Prefix(out NativeAttempt? __state)
        {
            var current = ActiveScreenContext.Instance.GetCurrentScreen();
            __state = GameStateService.CanConfirmSelection(current)
                ? Attempt("confirm_selection")
                : null;
        }
        static void Postfix(NativeAttempt? __state) =>
            FinishAttempt(__state);
    }

    [HarmonyPatch(typeof(NCardRewardSelectionScreen), "OnAlternateRewardSelected")]
    private static class RewardSkipPatch
    {
        static void Prefix(NCardRewardSelectionScreen __instance, int index,
            out NativeAttempt? __state)
        {
            var alternatives = GameStateService.GetCardRewardAlternatives(__instance);
            if (index < 0 || index >= alternatives.Count)
            {
                __state = null;
                CaptureGap("choose_reward_alternative",
                    "native reward alternative index is out of range");
                return;
            }

            __state = alternatives[index].OptionId.Equals(
                "Skip", StringComparison.OrdinalIgnoreCase)
                ? Attempt("skip_reward_cards")
                : Attempt("choose_reward_alternative", optionIndex: index);
        }
        static void Postfix(NativeAttempt? __state) => FinishAttempt(__state);
    }

    [HarmonyPatch(typeof(NEventRoom), nameof(NEventRoom.OptionButtonClicked))]
    private static class EventPatch
    {
        static void Prefix(int index, out Capture? __state) =>
            __state = Begin("choose_event_option", optionIndex: index);
        static void Postfix(Capture? __state) =>
            FinishOrGap("choose_event_option", __state);
    }

    [HarmonyPatch(typeof(NCrystalSphereScreen), "OnCellClicked")]
    private static class CrystalSpherePatch
    {
        static void Prefix(NCrystalSphereCell cell, out Capture? __state) =>
            __state = BeginCrystalSphereCell(cell);
        static void Postfix(Capture? __state) =>
            FinishOrGap("choose_crystal_sphere_cell", __state);
    }

    [HarmonyPatch(typeof(RestSiteSynchronizer),
        nameof(RestSiteSynchronizer.ChooseLocalOption))]
    private static class RestPatch
    {
        static void Prefix(int index, out Capture? __state) =>
            __state = Begin("choose_rest_option", optionIndex: index);
        static void Postfix(Capture? __state) =>
            FinishOrGap("choose_rest_option", __state);
    }

    [HarmonyPatch(typeof(NRewardButton), "OnRelease")]
    private static class RewardButtonReleasePatch
    {
        static void Prefix(NRewardButton __instance)
        {
            _pendingReward = __instance.Reward;
            _pendingRewardCapture = BeginRewardButton(__instance);
        }
    }

    [HarmonyPatch(typeof(RewardsSetSynchronizer),
        nameof(RewardsSetSynchronizer.SelectLocalReward))]
    private static class RewardPatch
    {
        static void Prefix(Reward reward, out NativeAttempt? __state)
        {
            __state = Volatile.Read(ref _suppressionDepth) != 0 ? null
                : new NativeAttempt("claim_reward",
                    TakePendingRewardCapture(reward) ?? BeginReward(reward));
        }
        static void Postfix(ref Task<bool> __result, NativeAttempt? __state)
        {
            if (__state != null)
                __result = FinishRewardIfTrue(__result, __state);
        }
    }

    [HarmonyPatch(typeof(MerchantEntry), nameof(MerchantEntry.OnTryPurchaseWrapper))]
    private static class MerchantPatch
    {
        static void Prefix(MerchantEntry __instance, out NativeAttempt? __state) =>
            __state = BeginMerchant(__instance);
        static void Postfix(ref Task<bool> __result, NativeAttempt? __state)
        { if (__state != null) __result = FinishMerchantIfTrue(__result, __state); }
    }

    [HarmonyPatch(typeof(MerchantCardRemovalEntry),
        nameof(MerchantCardRemovalEntry.OnTryPurchaseWrapper))]
    private static class RemovalPatch
    {
        static void Prefix(out Capture? __state) =>
            __state = Begin("remove_card_at_shop");
        static void Postfix(Capture? __state) =>
            FinishOrGap("remove_card_at_shop", __state);
    }

    [HarmonyPatch(typeof(NChooseABundleSelectionScreen), "OnBundleClicked")]
    private static class BundlePatch
    {
        static void Prefix(NCardBundle bundleNode, out Capture? __state)
        {
            var index = RefIndex(GameStateService.GetBundleOptions(
                ActiveScreenContext.Instance.GetCurrentScreen()), bundleNode);
            __state = index < 0 ? null : Begin("choose_bundle", optionIndex: index);
        }
        static void Postfix(Capture? __state) =>
            FinishOrGap("choose_bundle", __state);
    }
}
