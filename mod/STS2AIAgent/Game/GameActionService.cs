using Godot;
using System.Reflection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.DevConsole;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Debug;
using MegaCrit.Sts2.Core.Nodes.Debug.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.PauseMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline;
using MegaCrit.Sts2.Core.Nodes.Screens.Timeline.UnlockScreens;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.TopBar;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Multiplayer.Connection;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.Lobby;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Timeline;
using STS2AIAgent.Server;

namespace STS2AIAgent.Game;

internal static class GameActionService
{
    /// <summary>
    /// 记录智能体是否通过 skip_reward_cards 显式跳过卡牌奖励。
    /// 设置后，DrainRewardFlowAsync 不会自动领取卡牌；离开奖励页面时重置。
    /// </summary>
    private static bool _cardRewardSkipped;

    /// <summary>
    /// 回合内出牌计数。游戏内部计数无法通过反射访问，因此由 Mod 自行维护；
    /// 读取状态时与当前战斗回合同步，执行 play_card 时递增。
    /// </summary>
    internal static int CardsPlayedThisTurn { get; private set; }
    internal static int AttacksPlayedThisTurn { get; private set; }
    internal static int SkillsPlayedThisTurn { get; private set; }
    internal static int LastTurnNumber { get; private set; }

    /// <summary>
    /// resolve_rewards 设置该值后，TryResolveCardRewardAsync 会选择指定卡牌，
    /// 而不是默认第一项。-2 表示跳过，-1 表示没有待处理选择并使用默认行为。
    /// </summary>
    private static int _pendingCardRewardChoice = -1;

    /// <summary>开始新战斗时清空 Mod 自行维护的回合出牌计数。</summary>
    /// <param name="currentTurn">新战斗设置阶段的当前回合编号。</param>
    internal static void ResetCardPlayCounters(int currentTurn)
    {
        CardsPlayedThisTurn = 0;
        AttacksPlayedThisTurn = 0;
        SkillsPlayedThisTurn = 0;
        LastTurnNumber = currentTurn;
    }

    internal static void SyncCardPlayCounters(int currentTurn)
    {
        if (currentTurn == LastTurnNumber)
        {
            return;
        }

        CardsPlayedThisTurn = 0;
        AttacksPlayedThisTurn = 0;
        SkillsPlayedThisTurn = 0;
        LastTurnNumber = currentTurn;
    }

    /// <summary>在一次手动出牌成功后更新当前回合的分类计数。</summary>
    internal static void RecordCardPlayed(int currentTurn, string cardType)
    {
        SyncCardPlayCounters(currentTurn);
        CardsPlayedThisTurn++;
        if (cardType == "Attack") AttacksPlayedThisTurn++;
        else if (cardType == "Skill") SkillsPlayedThisTurn++;
    }

    public static Task<ActionResponsePayload> ExecuteAsync(ActionRequest request)
    {
        var actionName = request.action?.Trim().ToLowerInvariant();

        return actionName switch
        {
            "end_turn" => ExecuteEndTurnAsync(),
            "play_card" => ExecutePlayCardAsync(request),
            "continue_run" => ExecuteContinueRunAsync(),
            "abandon_run" => ExecuteAbandonRunAsync(),
            "save_and_quit" => ExecuteSaveAndQuitAsync(),
            "open_character_select" => ExecuteOpenCharacterSelectAsync(),
            "open_timeline" => ExecuteOpenTimelineAsync(),
            "close_main_menu_submenu" => ExecuteCloseMainMenuSubmenuAsync(),
            "choose_timeline_epoch" => ExecuteChooseTimelineEpochAsync(request),
            "confirm_timeline_overlay" => ExecuteConfirmTimelineOverlayAsync(),
            "choose_map_node" => ExecuteChooseMapNodeAsync(request),
            "claim_reward" => ExecuteClaimRewardAsync(request),
            "choose_reward_card" => ExecuteChooseRewardCardAsync(request),
            "skip_reward_cards" => ExecuteSkipRewardCardsAsync(),
            "choose_reward_alternative" => ExecuteChooseRewardAlternativeAsync(request),
            "select_deck_card" => ExecuteSelectDeckCardAsync(request),
            "skip_card_selection" => ExecuteSkipCardSelectionAsync(),
            "close_cards_view" => ExecuteCloseCardsViewAsync(),
            "confirm_selection" => ExecuteConfirmSelectionAsync(),
            "proceed" => ExecuteProceedAsync(),
            "open_chest" => ExecuteOpenChestAsync(),
            "choose_treasure_relic" => ExecuteChooseTreasureRelicAsync(request),
            "choose_event_option" => ExecuteChooseEventOptionAsync(request),
            "choose_crystal_sphere_cell" => ExecuteChooseCrystalSphereCellAsync(request),
            "choose_bundle" => ExecuteChooseBundleAsync(request),
            "confirm_bundle" => ExecuteConfirmBundleAsync(),
            "choose_rest_option" => ExecuteChooseRestOptionAsync(request),
            "open_shop_inventory" => ExecuteOpenShopInventoryAsync(),
            "close_shop_inventory" => ExecuteCloseShopInventoryAsync(),
            "buy_card" => ExecuteBuyCardAsync(request),
            "buy_relic" => ExecuteBuyRelicAsync(request),
            "buy_potion" => ExecuteBuyPotionAsync(request),
            "remove_card_at_shop" => ExecuteRemoveCardAtShopAsync(),
            "select_character" => ExecuteSelectCharacterAsync(request),
            "set_seed" => ExecuteSetSeedAsync(request),
            "embark" => ExecuteEmbarkAsync(),
            "unready" => ExecuteUnreadyAsync(),
            "host_multiplayer_lobby" => ExecuteHostMultiplayerLobbyAsync(),
            "join_multiplayer_lobby" => ExecuteJoinMultiplayerLobbyAsync(),
            "ready_multiplayer_lobby" => ExecuteReadyMultiplayerLobbyAsync(),
            "disconnect_multiplayer_lobby" => ExecuteDisconnectMultiplayerLobbyAsync(),
            "increase_ascension" => ExecuteAdjustAscensionAsync(1, "increase_ascension"),
            "decrease_ascension" => ExecuteAdjustAscensionAsync(-1, "decrease_ascension"),
            "use_potion" => ExecuteUsePotionAsync(request),
            "discard_potion" => ExecuteDiscardPotionAsync(request),
            "run_console_command" => ExecuteRunConsoleCommandAsync(request),
            "confirm_modal" => ExecuteConfirmModalAsync(),
            "dismiss_modal" => ExecuteDismissModalAsync(),
            "return_to_main_menu" => ExecuteReturnToMainMenuAsync(),
            _ => throw new ApiException(409, "invalid_action", "Action is not supported yet.", new
            {
                action = request.action
            })
        };
    }

    private static async Task<ActionResponsePayload> ExecuteEndTurnAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var combatState = CombatManager.Instance.DebugOnlyGetState();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanEndTurn(currentScreen, combatState, requireButtonReady: false))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "end_turn",
                screen
            });
        }

        var me = GameStateService.GetLocalPlayer(combatState)
            ?? throw new ApiException(503, "state_unavailable", "Local player is unavailable.", new
            {
                action = "end_turn",
                screen
            }, retryable: true);

        var playerCombatState = me.Creature.CombatState
            ?? throw new ApiException(503, "state_unavailable", "Combat state is unavailable.", new
            {
                action = "end_turn",
                screen
            }, retryable: true);

        var endTurnButton = GameStateService.GetEndTurnButton(currentScreen as NCombatRoom);
        if (!GameStateService.IsEndTurnButtonReady(endTurnButton))
        {
            throw new ApiException(503, "state_unavailable", "End turn button is unavailable.", new
            {
                action = "end_turn",
                screen
            }, retryable: true);
        }

        var roundNumber = playerCombatState.RoundNumber;
        endTurnButton!.ForceClick();

        var stable = await WaitForEndTurnTransitionAsync(roundNumber, TimeSpan.FromSeconds(5));

        return new ActionResponsePayload
        {
            action = "end_turn",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<bool> WaitForEndTurnTransitionAsync(int previousRound, TimeSpan timeout)
    {
        if (NGame.Instance == null)
        {
            return false;
        }

        var deadline = DateTime.UtcNow + timeout;

        while (DateTime.UtcNow < deadline)
        {
            await NGame.Instance.ToSignal(NGame.Instance.GetTree(), SceneTree.SignalName.ProcessFrame);

            if (IsEndTurnStable(previousRound))
            {
                return true;
            }
        }

        return IsEndTurnStable(previousRound);
    }

    private static bool IsEndTurnStable(int previousRound)
    {
        if (!CombatManager.Instance.IsInProgress)
        {
            return true;
        }

        var combatState = CombatManager.Instance.DebugOnlyGetState();
        if (combatState == null)
        {
            return true;
        }

        if (combatState.RoundNumber != previousRound)
        {
            return true;
        }

        if (combatState.CurrentSide != CombatSide.Player)
        {
            return true;
        }

        return !GameStateService.IsPlayerActionPhase(combatState);
    }

    private static async Task<ActionResponsePayload> ExecutePlayCardAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var combatState = CombatManager.Instance.DebugOnlyGetState();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanPlayAnyCard(currentScreen, combatState))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "play_card",
                screen
            });
        }

        if (request.card_index == null)
        {
            throw new ApiException(400, "invalid_request", "play_card requires card_index.", new
            {
                action = "play_card"
            });
        }

        var me = GameStateService.GetLocalPlayer(combatState)
            ?? throw new ApiException(503, "state_unavailable", "Local player is unavailable.", new
            {
                action = "play_card",
                screen
            }, retryable: true);

        var hand = me.PlayerCombatState?.Hand.Cards.ToList()
            ?? throw new ApiException(503, "state_unavailable", "Hand is unavailable.", new
            {
                action = "play_card",
                screen
            }, retryable: true);

        if (request.card_index < 0 || request.card_index >= hand.Count)
        {
            throw new ApiException(409, "invalid_target", "card_index is out of range.", new
            {
                action = "play_card",
                card_index = request.card_index,
                hand_count = hand.Count
            });
        }

        var card = hand[request.card_index.Value];
        if (!GameStateService.IsCardTargetSupported(card))
        {
            throw new ApiException(409, "invalid_action", "This target type is not supported by the API.", new
            {
                action = "play_card",
                card_index = request.card_index,
                card_id = card.Id.Entry,
                target_type = card.TargetType.ToString(),
                screen
            });
        }

        var target = ResolveCardTarget(request, combatState, card);

        if (!card.TryManualPlay(target))
        {
            throw new ApiException(409, "invalid_action", "Card cannot be played in the current state.", new
            {
                action = "play_card",
                card_index = request.card_index,
                target_index = request.target_index,
                card_id = card.Id.Entry,
                screen
            });
        }

        GameActionService.RecordCardPlayed(
            combatState?.RoundNumber ?? 0, card.Type.ToString());

        var stable = await WaitForPlayCardTransitionAsync(card, TimeSpan.FromSeconds(5));

        return new ActionResponsePayload
        {
            action = "play_card",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteOpenCharacterSelectAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (currentScreen is not NMainMenu mainMenu || !GameStateService.CanOpenCharacterSelect(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "open_character_select",
                screen
            });
        }

        var characterSelectScreen = mainMenu.SubmenuStack.GetSubmenuType<NCharacterSelectScreen>();
        characterSelectScreen.InitializeSingleplayer();
        mainMenu.SubmenuStack.Push(characterSelectScreen);
        var stable = await WaitForCharacterSelectOpenAsync(mainMenu, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "open_character_select",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteOpenTimelineAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (currentScreen is not NMainMenu mainMenu || !GameStateService.CanOpenTimeline(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "open_timeline",
                screen
            });
        }

        mainMenu.SubmenuStack.PushSubmenuType<NTimelineScreen>();
        var stable = await WaitForMainMenuSubmenuOpenAsync<NTimelineScreen>(mainMenu, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "open_timeline",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteCloseMainMenuSubmenuAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (currentScreen is not NSubmenu submenu || !GameStateService.CanCloseMainMenuSubmenu(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "close_main_menu_submenu",
                screen
            });
        }

        var submenuStack = GameStateService.GetMainMenuSubmenuStack(submenu)
            ?? throw new ApiException(503, "state_unavailable", "Main menu submenu stack is unavailable.", new
            {
                action = "close_main_menu_submenu",
                screen
            }, retryable: true);

        submenuStack.Pop();
        var stable = await WaitForMainMenuSubmenuCloseAsync(submenuStack, submenu, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "close_main_menu_submenu",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteChooseTimelineEpochAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseTimelineEpoch(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_timeline_epoch",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "option_index is required.", new
            {
                action = "choose_timeline_epoch"
            });
        }

        var slot = ResolveTimelineSlot(currentScreen, request.option_index.Value);
        var previousState = slot.State;

        slot.ForceClick();
        var stable = await WaitForTimelineEpochTransitionAsync(slot, previousState, TimeSpan.FromSeconds(15));

        return new ActionResponsePayload
        {
            action = "choose_timeline_epoch",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteConfirmTimelineOverlayAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanConfirmTimelineOverlay(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "confirm_timeline_overlay",
                screen
            });
        }

        var unlockScreen = GameStateService.GetTimelineUnlockScreen(currentScreen);
        if (unlockScreen != null)
        {
            var confirmButton = GameStateService.GetTimelineUnlockConfirmButton(currentScreen)
                ?? throw new ApiException(503, "state_unavailable", "Timeline unlock confirm button is unavailable.", new
                {
                    action = "confirm_timeline_overlay",
                    screen
                }, retryable: true);

            confirmButton.ForceClick();
            var unlockType = unlockScreen.GetType();
            var stable = await WaitForTimelineUnlockTransitionAsync(unlockType, TimeSpan.FromSeconds(10));

            return new ActionResponsePayload
            {
                action = "confirm_timeline_overlay",
                status = stable ? "completed" : "pending",
                stable = stable,
                message = stable ? "Action completed." : "Action queued but state is still transitioning.",
                state = GameStateService.BuildStatePayload()
            };
        }

        var closeButton = GameStateService.GetTimelineInspectCloseButton(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Timeline inspect close button is unavailable.", new
            {
                action = "confirm_timeline_overlay",
                screen
            }, retryable: true);

        closeButton.ForceClick();
        var inspectScreen = GameStateService.GetTimelineInspectScreen(currentScreen);
        var stableInspect = await WaitForTimelineInspectCloseAsync(inspectScreen, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "confirm_timeline_overlay",
            status = stableInspect ? "completed" : "pending",
            stable = stableInspect,
            message = stableInspect ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteContinueRunAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (currentScreen is not NMainMenu mainMenu || !GameStateService.CanContinueRun(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "continue_run",
                screen
            });
        }

        var continueButton = GameStateService.GetMainMenuContinueButton(mainMenu)
            ?? throw new ApiException(503, "state_unavailable", "Continue button is unavailable.", new
            {
                action = "continue_run",
                screen
            }, retryable: true);

        continueButton.ForceClick();
        var stable = await WaitForMainMenuExitAsync(mainMenu, TimeSpan.FromSeconds(15));

        return new ActionResponsePayload
        {
            action = "continue_run",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteAbandonRunAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (currentScreen is not NMainMenu mainMenu || !GameStateService.CanAbandonRun(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "abandon_run",
                screen
            });
        }

        var abandonButton = GameStateService.GetMainMenuAbandonRunButton(mainMenu)
            ?? throw new ApiException(503, "state_unavailable", "Abandon run button is unavailable.", new
            {
                action = "abandon_run",
                screen
            }, retryable: true);

        abandonButton.ForceClick();
        var stable = await WaitForMainMenuModalAsync(TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "abandon_run",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteSaveAndQuitAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var runState = RunManager.Instance.DebugOnlyGetState();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanSaveAndQuit(currentScreen, runState))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "save_and_quit",
                screen
            });
        }

        var pauseMenu = FindPauseMenu();
        if (pauseMenu == null)
        {
            var pauseButton = FindFirstInGame<NTopBarPauseButton>();
            if (pauseButton == null || !pauseButton.IsVisibleInTree())
            {
                throw new ApiException(503, "state_unavailable", "Pause button is unavailable.", new
                {
                    action = "save_and_quit",
                    screen
                }, retryable: true);
            }

            pauseButton.ForceClick();
            pauseMenu = await WaitForPauseMenuAsync(TimeSpan.FromSeconds(5));
        }

        if (pauseMenu == null)
        {
            throw new ApiException(503, "state_unavailable", "Pause menu did not open.", new
            {
                action = "save_and_quit",
                screen
            }, retryable: true);
        }

        var closeTask = InvokePrivateTask(pauseMenu, "CloseToMenu");
        if (closeTask != null)
        {
            await closeTask;
        }
        else
        {
            var saveAndQuitButton = GetPrivateField<NButton>(pauseMenu, "_saveAndQuitButton");
            if (saveAndQuitButton == null || !saveAndQuitButton.IsVisibleInTree() || !saveAndQuitButton.IsEnabled)
            {
                throw new ApiException(503, "state_unavailable", "Save and Quit button is unavailable.", new
                {
                    action = "save_and_quit",
                    screen
                }, retryable: true);
            }

            saveAndQuitButton.ForceClick();
        }

        var stable = await WaitForMainMenuAfterSaveAndQuitAsync(TimeSpan.FromSeconds(20));

        return new ActionResponsePayload
        {
            action = "save_and_quit",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static Creature? ResolveCardTarget(ActionRequest request, CombatState? combatState, CardModel card)
    {
        if (!GameStateService.CardRequiresTarget(card))
        {
            return null;
        }

        if (combatState == null)
        {
            throw new ApiException(503, "state_unavailable", "Combat state is unavailable.", new
            {
                action = "play_card",
                card_id = card.Id.Entry
            }, retryable: true);
        }

        if (request.target_index == null)
        {
            throw new ApiException(409, "invalid_target", "This card requires target_index.", new
            {
                action = "play_card",
                card_id = card.Id.Entry,
                target_type = card.TargetType.ToString(),
                target_index_space = card.TargetType == TargetType.AnyEnemy ? "enemies" : "players"
            });
        }

        if (card.TargetType == TargetType.AnyEnemy)
        {
            var enemy = GameStateService.ResolveEnemyTarget(combatState, request.target_index.Value);
            if (enemy == null)
            {
                throw new ApiException(409, "invalid_target", "target_index is out of range for combat.enemies[].", new
                {
                    action = "play_card",
                    card_id = card.Id.Entry,
                    target_index = request.target_index,
                    target_index_space = "enemies"
                });
            }

            return enemy;
        }

        if (card.TargetType == TargetType.AnyAlly)
        {
            var allyTargetIndices = GameStateService.GetTargetablePlayerIndices(combatState, card.Owner, allowSelf: false);
            if (!allyTargetIndices.Contains(request.target_index.Value))
            {
                throw new ApiException(409, "invalid_target", "target_index is out of range for combat.players[].", new
                {
                    action = "play_card",
                    card_id = card.Id.Entry,
                    target_index = request.target_index,
                    target_index_space = "players"
                });
            }

            return GameStateService.ResolvePlayerTarget(combatState, request.target_index.Value);
        }

        throw new ApiException(409, "invalid_action", "This target type is not supported yet.", new
        {
            action = "play_card",
            card_id = card.Id.Entry,
            target_type = card.TargetType.ToString()
        });
    }

    private static async Task<bool> WaitForPlayCardTransitionAsync(CardModel card, TimeSpan timeout)
    {
        if (NGame.Instance == null)
        {
            return false;
        }

        var deadline = DateTime.UtcNow + timeout;

        while (DateTime.UtcNow < deadline)
        {
            await NGame.Instance.ToSignal(NGame.Instance.GetTree(), SceneTree.SignalName.ProcessFrame);

            if (IsPlayCardStable(card))
            {
                return true;
            }

            if (IsPlayCardAwaitingPlayerInput())
            {
                return false;
            }
        }

        return IsPlayCardStable(card);
    }

    private static bool IsPlayCardStable(CardModel card)
    {
        if (!CombatManager.Instance.IsInProgress)
        {
            return true;
        }

        if (card.Pile?.Type == PileType.Hand)
        {
            return false;
        }

        return ArePlayerDrivenActionsSettled();
    }

    private static bool IsPlayCardAwaitingPlayerInput()
    {
        if (!CombatManager.Instance.IsInProgress)
        {
            return false;
        }

        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        return currentScreen != null && GameStateService.ResolveScreen(currentScreen) == "CARD_SELECTION";
    }

    private static bool ArePlayerDrivenActionsSettled()
    {
        var runningAction = RunManager.Instance.ActionExecutor.CurrentlyRunningAction;
        if (runningAction != null && ActionQueueSet.IsGameActionPlayerDriven(runningAction))
        {
            return false;
        }

        var readyAction = RunManager.Instance.ActionQueueSet.GetReadyAction();
        if (readyAction != null && ActionQueueSet.IsGameActionPlayerDriven(readyAction))
        {
            return false;
        }

        return true;
    }

    internal static bool AreGameActionsSettled()
    {
        if (RunManager.Instance.ActionExecutor.CurrentlyRunningAction != null)
        {
            return false;
        }

        return RunManager.Instance.ActionQueueSet.GetReadyAction() == null;
    }

    private static async Task<ActionResponsePayload> ExecuteChooseMapNodeAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var runState = RunManager.Instance.DebugOnlyGetState();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseMapNode(currentScreen, runState))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_map_node",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "choose_map_node requires option_index.", new
            {
                action = "choose_map_node"
            });
        }

        var availableNodes = GameStateService.GetAvailableMapNodes(currentScreen, runState);
        if (request.option_index < 0 || request.option_index >= availableNodes.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_map_node",
                option_index = request.option_index,
                node_count = availableNodes.Count
            });
        }

        var selectedNode = availableNodes[request.option_index.Value];
        var roomEntered = false;
        var isMultiplayerVote = runState != null && RunManager.Instance.NetService.Type.IsMultiplayer() && runState.Players.Count > 1;

        void OnRoomEntered()
        {
            roomEntered = true;
        }

        RunManager.Instance.RoomEntered += OnRoomEntered;
        try
        {
            if (isMultiplayerVote)
            {
                var mapScreen = NMapScreen.Instance
                    ?? currentScreen as NMapScreen
                    ?? throw new ApiException(503, "state_unavailable", "Map screen is unavailable.", new
                    {
                        action = "choose_map_node",
                        screen
                    }, retryable: true);

                mapScreen.OnMapPointSelectedLocally(selectedNode);
            }
            else
            {
                selectedNode.ForceClick();
            }

            var stable = isMultiplayerVote
                ? await WaitForMultiplayerMapVoteOrTransitionAsync(selectedNode.Point.coord, TimeSpan.FromSeconds(10), () => roomEntered)
                : await WaitForMapTransitionAsync(TimeSpan.FromSeconds(10), () => roomEntered);
            var roomStarted = HasEnteredMapDestination(() => roomEntered);

            return new ActionResponsePayload
            {
                action = "choose_map_node",
                status = stable ? "completed" : "pending",
                stable = stable,
                message = stable
                    ? roomStarted
                        ? "Action completed."
                        : "Map vote submitted. Waiting for other players to finish choosing."
                    : "Action queued but state is still transitioning.",
                state = GameStateService.BuildStatePayload()
            };
        }
        finally
        {
            RunManager.Instance.RoomEntered -= OnRoomEntered;
        }
    }

    private static async Task<bool> WaitForMultiplayerMapVoteOrTransitionAsync(
        MapCoord targetCoord,
        TimeSpan timeout,
        Func<bool> roomEntered)
    {
        if (NGame.Instance == null)
        {
            return false;
        }

        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await NGame.Instance.ToSignal(NGame.Instance.GetTree(), SceneTree.SignalName.ProcessFrame);

            if (HasEnteredMapDestination(roomEntered) || IsLocalMapVoteRegistered(targetCoord))
            {
                return true;
            }
        }

        return HasEnteredMapDestination(roomEntered) || IsLocalMapVoteRegistered(targetCoord);
    }

    private static async Task<bool> WaitForMapTransitionAsync(TimeSpan timeout, Func<bool> roomEntered)
    {
        if (NGame.Instance == null)
        {
            return false;
        }

        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await NGame.Instance.ToSignal(NGame.Instance.GetTree(), SceneTree.SignalName.ProcessFrame);

            if (IsMapTransitionStable(roomEntered))
            {
                return true;
            }
        }

        return IsMapTransitionStable(roomEntered);
    }

    private static bool IsMapTransitionStable(Func<bool> roomEntered)
    {
        if (!HasEnteredMapDestination(roomEntered))
        {
            return false;
        }

        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var runState = RunManager.Instance.DebugOnlyGetState();
        if (!DoesScreenMatchCurrentRoom(currentScreen, runState?.CurrentRoom))
        {
            return false;
        }

        return IsStableScreenState(currentScreen, allowMapScreen: false);
    }

    private static bool HasEnteredMapDestination(Func<bool> roomEntered)
    {
        if (roomEntered())
        {
            return true;
        }

        var runState = RunManager.Instance.DebugOnlyGetState();
        return runState?.CurrentRoom is not null && runState.CurrentRoom is not MapRoom;
    }

    private static bool IsLocalMapVoteRegistered(MapCoord targetCoord)
    {
        var runState = RunManager.Instance.DebugOnlyGetState();
        var localPlayer = GameStateService.GetLocalPlayer(runState);
        if (runState == null || localPlayer == null)
        {
            return false;
        }

        var vote = RunManager.Instance.MapSelectionSynchronizer.GetVote(localPlayer);
        return vote.HasValue &&
            vote.Value.coord.row == targetCoord.row &&
            vote.Value.coord.col == targetCoord.col;
    }

    private static bool DoesScreenMatchCurrentRoom(IScreenContext? currentScreen, AbstractRoom? currentRoom)
    {
        if (currentRoom == null)
        {
            return false;
        }

        var screen = GameStateService.ResolveScreen(currentScreen);
        return currentRoom switch
        {
            CombatRoom => screen == "COMBAT",
            EventRoom => screen == "EVENT",
            MerchantRoom => screen == "SHOP",
            RestSiteRoom => screen == "REST",
            TreasureRoom => screen == "CHEST",
            MapRoom => screen == "MAP",
            _ => screen != "UNKNOWN" && screen != "MAP"
        };
    }

    private static bool IsStableScreenState(IScreenContext? currentScreen, bool allowMapScreen)
    {
        var screen = GameStateService.ResolveScreen(currentScreen);
        if (screen == "UNKNOWN")
        {
            return false;
        }

        if (screen == "COMBAT")
        {
            return currentScreen is NCombatRoom combatRoom &&
                combatRoom.Mode == CombatRoomMode.ActiveCombat &&
                CombatManager.Instance.IsInProgress &&
                !CombatManager.Instance.IsOverOrEnding &&
                GameStateService.IsPlayerActionPhase(CombatManager.Instance.DebugOnlyGetState()) &&
                !CombatManager.Instance.PlayerActionsDisabled &&
                CombatManager.Instance.DebugOnlyGetState() != null;
        }

        if (screen != "MAP")
        {
            return true;
        }

        if (!allowMapScreen)
        {
            return false;
        }

        return currentScreen is NMapScreen mapScreen && !mapScreen.IsTraveling;
    }

    private static async Task<ActionResponsePayload> ExecuteProceedAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanProceed(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "proceed",
                screen
            });
        }

        var proceedButton = GameStateService.GetProceedButton(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Proceed button not found.", new
            {
                action = "proceed",
                screen
            }, retryable: true);

        proceedButton.ForceClick();
        var stable = await WaitForProceedTransitionAsync(currentScreen, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "proceed",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<bool> WaitForProceedTransitionAsync(
        IScreenContext? previousScreen,
        TimeSpan timeout)
    {
        if (NGame.Instance == null)
        {
            return false;
        }

        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (IsProceedStable(previousScreen))
            {
                return true;
            }
        }

        return IsProceedStable(previousScreen);
    }

    private static bool IsProceedStable(IScreenContext? previousScreen)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        if (ReferenceEquals(currentScreen, previousScreen))
        {
            return false;
        }

        return IsStableScreenState(currentScreen, allowMapScreen: true);
    }

    private static async Task<ActionResponsePayload> ExecuteResolveRewardsAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanCollectRewardsAndProceed(currentScreen) &&
            currentScreen is not NCardRewardSelectionScreen)
        {
            throw new ApiException(409, "invalid_action", "Not on reward screen.", new
            {
                action = "resolve_rewards",
                screen
            });
        }

        // option_index：-1 表示跳过，0/1/2 表示选择对应卡牌。字段缺失时保留
        // skip_reward_cards 的显式跳过状态，否则使用自动处理。card_index 作为
        // 选择卡牌的向后兼容别名继续接受。
        if (request.option_index.HasValue)
        {
            if (request.option_index.Value == -1)
            {
                _pendingCardRewardChoice = -2;
                _cardRewardSkipped = true;
            }
            else
            {
                _pendingCardRewardChoice = request.option_index.Value;
                _cardRewardSkipped = false;
            }
        }
        else if (request.card_index.HasValue)
        {
            _pendingCardRewardChoice = request.card_index.Value;
            _cardRewardSkipped = false;
        }
        else
        {
            _pendingCardRewardChoice = -1;
        }

        var stable = await DrainRewardFlowAsync(TimeSpan.FromSeconds(20));

        return new ActionResponsePayload
        {
            action = "resolve_rewards",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "All rewards resolved." : "Reward flow still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteCollectRewardsAndProceedAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanCollectRewardsAndProceed(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "collect_rewards_and_proceed",
                screen
            });
        }

        var stable = await DrainRewardFlowAsync(TimeSpan.FromSeconds(20));

        return new ActionResponsePayload
        {
            action = "collect_rewards_and_proceed",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Reward flow is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteClaimRewardAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanClaimReward(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "claim_reward",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "claim_reward requires option_index.", new
            {
                action = "claim_reward"
            });
        }

        var rewardButtons = GameStateService.GetRewardButtons(currentScreen);

        if (request.option_index < 0 || request.option_index >= rewardButtons.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "claim_reward",
                option_index = request.option_index,
                option_count = rewardButtons.Count
            });
        }

        var selectedReward = rewardButtons[request.option_index.Value];
        if (!GameStateService.IsRewardClaimable(selectedReward))
        {
            throw new ApiException(409, "invalid_action", "The selected reward is not claimable in the current state.", new
            {
                action = "claim_reward",
                option_index = request.option_index
            });
        }

        var previousRewardCount = rewardButtons.Count(button => button.IsEnabled);
        selectedReward.ForceClick();
        var stable = await WaitForRewardButtonResolutionAsync(currentScreen, previousRewardCount, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "claim_reward",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteChooseRewardCardAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseRewardCard(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_reward_card",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "choose_reward_card requires option_index.", new
            {
                action = "choose_reward_card"
            });
        }

        var options = GameStateService.GetCardRewardOptions(currentScreen);
        if (request.option_index < 0 || request.option_index >= options.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_reward_card",
                option_index = request.option_index,
                option_count = options.Count
            });
        }

        var selected = options[request.option_index.Value];
        var previousOptionCount = options.Count;
        selected.EmitSignal(NCardHolder.SignalName.Pressed, selected);
        _cardRewardSkipped = false; // 已领取卡牌，清除先前的跳过状态。
        var stable = await WaitForRewardCardResolutionAsync(currentScreen, previousOptionCount, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "choose_reward_card",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteSkipRewardCardsAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanSkipRewardCards(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "skip_reward_cards",
                screen
            });
        }

        var alternatives = GameStateService.GetCardRewardAlternatives(currentScreen);
        var selectedIndex = alternatives
            .Select((option, index) => (option, index))
            .First(pair => pair.option.OptionId.Equals(
                "Skip", StringComparison.OrdinalIgnoreCase)).index;
        SelectCardRewardAlternative((NCardRewardSelectionScreen)currentScreen!,
            selectedIndex);
        _cardRewardSkipped = true;
        var stable = await WaitForRewardCardResolutionAsync(currentScreen, GameStateService.GetCardRewardOptions(currentScreen).Count, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "skip_reward_cards",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteSelectDeckCardAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanSelectDeckCard(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "select_deck_card",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "select_deck_card requires option_index.", new
            {
                action = "select_deck_card"
            });
        }

        var options = GameStateService.GetDeckSelectionOptions(currentScreen);
        if (request.option_index < 0 || request.option_index >= options.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "select_deck_card",
                option_index = request.option_index,
                option_count = options.Count
            });
        }

        var isCombatHandSelection = GameStateService.TryGetCombatHandSelectionMetadata(currentScreen, out var combatHand, out var combatHandSelection);
        var isGridSelection = GameStateService.TryGetGridSelectionMetadata(
            currentScreen, out var gridSelectionScreen, out var gridSelection);
        var selected = options[request.option_index.Value];
        if (isGridSelection &&
            gridSelection.MaxSelect > 0 &&
            gridSelection.SelectedCards.Count >= gridSelection.MaxSelect &&
            selected.CardModel != null &&
            !gridSelection.SelectedCards.Contains(selected.CardModel))
        {
            throw new ApiException(409, "invalid_action",
                "The maximum number of cards is already selected.", new
                {
                    action = "select_deck_card",
                    selected_count = gridSelection.SelectedCards.Count,
                    max_select = gridSelection.MaxSelect
                });
        }
        if (isCombatHandSelection)
        {
            if (selected is not NHandCardHolder handHolder)
            {
                throw new ApiException(503, "state_unavailable", "Combat hand selection holder is unavailable.", new
                {
                    action = "select_deck_card",
                    screen
                }, retryable: true);
            }

            combatHand!.Call(
                combatHand.CurrentMode == NPlayerHand.Mode.UpgradeSelect
                    ? NPlayerHand.MethodName.SelectCardInUpgradeMode
                    : NPlayerHand.MethodName.SelectCardInSimpleMode,
                handHolder);
            combatHand.Call(NPlayerHand.MethodName.CheckIfSelectionComplete);
        }
        else
        {
            selected.EmitSignal(NCardHolder.SignalName.Pressed, selected);
        }

        var stable = currentScreen switch
        {
            NCardGridSelectionScreen when isGridSelection && gridSelectionScreen != null =>
                gridSelection.MaxSelect <= 1
                    ? await ConfirmDeckSelectionAsync(gridSelectionScreen, TimeSpan.FromSeconds(10))
                    : await WaitForGridSelectionStepAsync(
                        gridSelectionScreen, gridSelection, TimeSpan.FromSeconds(2)),
            NChooseACardSelectionScreen chooseCardScreen => await WaitForChooseCardSelectionResolutionAsync(chooseCardScreen, TimeSpan.FromSeconds(10)),
            _ when isCombatHandSelection => await WaitForCombatHandSelectionStepAsync(combatHandSelection, TimeSpan.FromSeconds(10)),
            _ => false
        };

        return new ActionResponsePayload
        {
            action = "select_deck_card",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteSkipCardSelectionAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var button = GameStateService.GetCardSelectionSkipButton(currentScreen);
        if (button == null)
        {
            throw new ApiException(409, "invalid_action",
                "Action is not available in the current state.", new
                {
                    action = "skip_card_selection",
                    screen
                });
        }

        button.ForceClick();
        var stable = currentScreen is NChooseACardSelectionScreen chooseCardScreen &&
            await WaitForChooseCardSelectionResolutionAsync(
                chooseCardScreen, TimeSpan.FromSeconds(10));
        return new ActionResponsePayload
        {
            action = "skip_card_selection",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." :
                "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteChooseRewardAlternativeAsync(
        ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseRewardAlternative(currentScreen))
        {
            throw new ApiException(409, "invalid_action",
                "Action is not available in the current state.", new
                {
                    action = "choose_reward_alternative",
                    screen
                });
        }
        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request",
                "choose_reward_alternative requires option_index.", new
                {
                    action = "choose_reward_alternative"
                });
        }

        var alternatives = GameStateService.GetCardRewardAlternatives(currentScreen);
        if (request.option_index < 0 || request.option_index >= alternatives.Count)
        {
            throw new ApiException(409, "invalid_target",
                "option_index is out of range.", new
                {
                    action = "choose_reward_alternative",
                    option_index = request.option_index,
                    option_count = alternatives.Count
                });
        }
        var selected = alternatives[request.option_index.Value];
        if (selected.OptionId.Equals("Skip", StringComparison.OrdinalIgnoreCase))
        {
            throw new ApiException(409, "invalid_target",
                "Use skip_reward_cards for the Skip alternative.", new
                {
                    action = "choose_reward_alternative",
                    option_index = request.option_index,
                    option_id = selected.OptionId
                });
        }

        var previousOptions = GameStateService.GetCardRewardOptions(currentScreen)
            .ToArray();
        SelectCardRewardAlternative((NCardRewardSelectionScreen)currentScreen!,
            request.option_index.Value);
        _cardRewardSkipped = false;
        var stable = await WaitForRewardAlternativeResolutionAsync(
            currentScreen, previousOptions, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "choose_reward_alternative",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." :
                "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static void SelectCardRewardAlternative(
        NCardRewardSelectionScreen screen, int index)
    {
        typeof(NCardRewardSelectionScreen)
            .GetMethod("OnAlternateRewardSelected",
                BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(screen, new object[] { index });
    }

    private static async Task<bool> WaitForRewardAlternativeResolutionAsync(
        IScreenContext? previousScreen,
        IReadOnlyList<NCardHolder> previousOptions,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!ReferenceEquals(currentScreen, previousScreen))
            {
                return true;
            }

            var currentOptions = GameStateService.GetCardRewardOptions(currentScreen);
            if (currentOptions.Count != previousOptions.Count ||
                currentOptions.Where((holder, index) =>
                    !ReferenceEquals(holder, previousOptions[index])).Any())
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<ActionResponsePayload> ExecuteConfirmSelectionAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanConfirmSelection(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "confirm_selection",
                screen
            });
        }

        bool stable;
        if (currentScreen is NCardGridSelectionScreen gridSelection)
        {
            stable = await ConfirmDeckSelectionAsync(
                gridSelection, TimeSpan.FromSeconds(10));
        }
        else if (GameStateService.TryGetCombatHandSelection(
                     currentScreen, out var combatHand) &&
                 combatHand != null &&
                 TryGetCombatHandConfirmButton(combatHand, out var confirmButton) &&
                 confirmButton != null)
        {
            confirmButton.ForceClick();
            stable = await WaitForCombatHandSelectionResolutionAsync(
                TimeSpan.FromSeconds(10));
        }
        else
        {
            throw new ApiException(409, "invalid_action",
                "Selection confirmation control is unavailable.", new
                {
                    action = "confirm_selection",
                    screen
                });
        }

        return new ActionResponsePayload
        {
            action = "confirm_selection",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteCloseCardsViewAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanCloseCardsView(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "close_cards_view",
                screen
            });
        }

        var backButton = GameStateService.GetCardsViewBackButton(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Cards view back button is unavailable.", new
            {
                action = "close_cards_view",
                screen
            }, retryable: true);

        backButton.ForceClick();
        var stable = await WaitForCardsViewCloseAsync(TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "close_cards_view",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<bool> WaitForChooseCardSelectionResolutionAsync(
        NChooseACardSelectionScreen selectionScreen,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is not NChooseACardSelectionScreen || !GodotObject.IsInstanceValid(selectionScreen))
            {
                return true;
            }
        }

        return ActiveScreenContext.Instance.GetCurrentScreen() is not NChooseACardSelectionScreen;
    }

    private static async Task<bool> WaitForCardsViewCloseAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (ActiveScreenContext.Instance.GetCurrentScreen() is not NCardsViewScreen)
            {
                return true;
            }
        }

        return ActiveScreenContext.Instance.GetCurrentScreen() is not NCardsViewScreen;
    }

    private static async Task<bool> WaitForCombatHandSelectionResolutionAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!GameStateService.TryGetCombatHandSelection(currentScreen, out var currentHand) ||
                currentHand == null ||
                !GodotObject.IsInstanceValid(currentHand))
            {
                return true;
            }
        }

        return !GameStateService.TryGetCombatHandSelection(ActiveScreenContext.Instance.GetCurrentScreen(), out _);
    }

    private static async Task<bool> WaitForCombatHandSelectionStepAsync(
        CombatHandSelectionMetadata previousSelection,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!GameStateService.TryGetCombatHandSelectionMetadata(currentScreen, out _, out var currentSelection))
            {
                return true;
            }

            if (currentSelection.SelectedCount != previousSelection.SelectedCount)
            {
                if (!currentSelection.RequiresConfirmation &&
                    currentSelection.SelectedCount >= currentSelection.MaxSelect)
                {
                    continue;
                }

                return false;
            }
        }

        return !GameStateService.TryGetCombatHandSelection(ActiveScreenContext.Instance.GetCurrentScreen(), out _);
    }

    private static bool TryGetCombatHandConfirmButton(NPlayerHand hand, out NConfirmButton? confirmButton)
    {
        confirmButton = hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton")
            ?? hand.GetNodeOrNull<NConfirmButton>("SelectModeConfirmButton");
        return confirmButton != null && GodotObject.IsInstanceValid(confirmButton);
    }

    private static async Task<bool> DrainRewardFlowAsync(TimeSpan timeout)
    {
        if (NGame.Instance == null)
        {
            return false;
        }

        var deadline = DateTime.UtcNow + timeout;
        var attemptedRewardButtons = new HashSet<ulong>();

        while (DateTime.UtcNow < deadline)
        {
            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();

            if (currentScreen is NCardRewardSelectionScreen cardRewardScreen)
            {
                if (!await TryResolveCardRewardAsync(cardRewardScreen, deadline))
                {
                    return false;
                }

                continue;
            }

            if (currentScreen is not NRewardsScreen rewardsScreen)
            {
                _cardRewardSkipped = false;
                return true;
            }

            if (TryGetNextClaimableRewardButton(rewardsScreen, attemptedRewardButtons, out var rewardButton))
            {
                attemptedRewardButtons.Add(rewardButton!.GetInstanceId());
                await ClickRewardButtonAsync(rewardButton, deadline);
                continue;
            }

            var proceedButton = GameStateService.GetRewardProceedButton(rewardsScreen);
            if (proceedButton != null && proceedButton.IsEnabled)
            {
                proceedButton.ForceClick();
                return await WaitForRewardFlowExitAsync(rewardsScreen, deadline);
            }

            return IsRewardFlowStable();
        }

        return IsRewardFlowStable();
    }

    private static bool TryGetNextClaimableRewardButton(
        NRewardsScreen rewardsScreen,
        HashSet<ulong> attemptedRewardButtons,
        out NRewardButton? rewardButton)
    {
        rewardButton = GameStateService
            .GetRewardButtons(rewardsScreen)
            .FirstOrDefault(button =>
                GameStateService.IsRewardClaimable(button) &&
                !attemptedRewardButtons.Contains(button.GetInstanceId()) &&
                (!_cardRewardSkipped || button.Reward is not CardReward));

        return rewardButton != null;
    }

    private static async Task ClickRewardButtonAsync(NRewardButton rewardButton, DateTime deadline)
    {
        var previousRewardCount = GameStateService.GetRewardButtons(ActiveScreenContext.Instance.GetCurrentScreen()).Count;
        rewardButton.ForceClick();

        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is NCardRewardSelectionScreen)
            {
                return;
            }

            var rewardButtons = GameStateService.GetRewardButtons(currentScreen);
            if (!GodotObject.IsInstanceValid(rewardButton) || rewardButtons.Count != previousRewardCount)
            {
                return;
            }
        }
    }

    private static async Task<bool> TryResolveCardRewardAsync(NCardRewardSelectionScreen cardRewardScreen, DateTime deadline)
    {
        for (var i = 0; i < 24 && DateTime.UtcNow < deadline; i++)
        {
            await WaitForNextFrameAsync();
        }

        // resolve_rewards 请求跳过时，点击跳过选项。
        if (_pendingCardRewardChoice == -2)
        {
            _pendingCardRewardChoice = -1;
            var alternatives = GameStateService.GetCardRewardAlternatives(cardRewardScreen);
            var skipIndex = alternatives
                .Select((option, index) => (option, index))
                .FirstOrDefault(pair => pair.option.OptionId.Equals(
                    "Skip", StringComparison.OrdinalIgnoreCase)).index;
            if (alternatives.Count > 0 &&
                alternatives[skipIndex].OptionId.Equals(
                    "Skip", StringComparison.OrdinalIgnoreCase))
            {
                SelectCardRewardAlternative(cardRewardScreen, skipIndex);
                _cardRewardSkipped = true;
            }
            while (DateTime.UtcNow < deadline)
            {
                await WaitForNextFrameAsync();
                if (!GodotObject.IsInstanceValid(cardRewardScreen) ||
                    ActiveScreenContext.Instance.GetCurrentScreen() is not NCardRewardSelectionScreen)
                    return true;
            }
            return false;
        }

        var options = GameStateService.GetCardRewardOptions(cardRewardScreen);

        // resolve_rewards 指定卡牌索引时，按该索引选择。
        NCardHolder? selected;
        if (_pendingCardRewardChoice >= 0 && _pendingCardRewardChoice < options.Count)
        {
            selected = options[_pendingCardRewardChoice];
            _pendingCardRewardChoice = -1;
        }
        else
        {
            selected = options.FirstOrDefault();
        }
        if (selected == null)
        {
            return false;
        }

        selected.EmitSignal(NCardHolder.SignalName.Pressed, selected);
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (!GodotObject.IsInstanceValid(cardRewardScreen) ||
                ActiveScreenContext.Instance.GetCurrentScreen() is not NCardRewardSelectionScreen)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<bool> WaitForRewardFlowExitAsync(NRewardsScreen rewardsScreen, DateTime deadline)
    {
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (!GodotObject.IsInstanceValid(rewardsScreen))
            {
                return true;
            }

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen != rewardsScreen)
            {
                return true;
            }

            if (NOverlayStack.Instance?.Peek() != rewardsScreen)
            {
                return true;
            }
        }

        return IsRewardFlowStable();
    }

    private static bool IsRewardFlowStable()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        return currentScreen is not NRewardsScreen && currentScreen is not NCardRewardSelectionScreen;
    }

    private static async Task<bool> WaitForRewardCardResolutionAsync(
        IScreenContext? previousScreen,
        int previousOptionCount,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!ReferenceEquals(currentScreen, previousScreen))
            {
                return true;
            }

            if (GameStateService.GetCardRewardOptions(currentScreen).Count != previousOptionCount)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<bool> WaitForRewardButtonResolutionAsync(
        IScreenContext? previousScreen,
        int previousRewardCount,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!ReferenceEquals(currentScreen, previousScreen))
            {
                return true;
            }

            var currentRewardCount = GameStateService.GetRewardButtons(currentScreen).Count(button => button.IsEnabled);
            if (currentRewardCount != previousRewardCount)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<bool> ConfirmDeckSelectionAsync(NCardGridSelectionScreen screen, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;

        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (!GodotObject.IsInstanceValid(screen))
            {
                return true;
            }

            var previewContainer = screen.GetNodeOrNull<Control>("%PreviewContainer");
            var previewConfirm = screen.GetNodeOrNull<NConfirmButton>("%PreviewConfirm")
                ?? previewContainer?.GetNodeOrNull<NConfirmButton>("Confirm");
            if (previewContainer?.Visible == true && previewConfirm?.IsEnabled == true)
            {
                previewConfirm.ForceClick();
                return await WaitForDeckSelectionResolutionAsync(screen, deadline);
            }

            if (screen is NDeckTransformSelectScreen transformScreen &&
                TryGetDeckTransformConfirmButton(transformScreen, out var transformConfirm))
            {
                transformConfirm!.ForceClick();
                return await WaitForDeckSelectionResolutionAsync(screen, deadline);
            }

            if (screen is NDeckEnchantSelectScreen enchantScreen &&
                TryGetDeckEnchantConfirmButton(enchantScreen, out var enchantConfirm))
            {
                enchantConfirm!.ForceClick();
                return await WaitForDeckSelectionResolutionAsync(screen, deadline);
            }

            if (screen is NDeckUpgradeSelectScreen upgradeScreen &&
                TryGetDeckUpgradeConfirmButton(upgradeScreen, out var upgradeConfirm))
            {
                upgradeConfirm!.ForceClick();
                return await WaitForDeckSelectionResolutionAsync(screen, deadline);
            }

            var confirmButton = screen.GetNodeOrNull<NConfirmButton>("%Confirm")
                ?? screen.GetNodeOrNull<NConfirmButton>("Confirm");
            if (confirmButton?.IsEnabled == true)
            {
                confirmButton.ForceClick();
            }
        }

        return false;
    }

    private static async Task<bool> WaitForGridSelectionStepAsync(
        NCardGridSelectionScreen screen,
        GridSelectionMetadata previousSelection,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();
            var current = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!GodotObject.IsInstanceValid(screen) ||
                !ReferenceEquals(current, screen))
            {
                return true;
            }

            if (GameStateService.TryGetGridSelectionMetadata(
                    current, out _, out var selection) &&
                !selection.SelectedCards.SetEquals(previousSelection.SelectedCards))
            {
                return true;
            }
        }

        return false;
    }

    private static bool TryGetDeckUpgradeConfirmButton(
        NDeckUpgradeSelectScreen screen,
        out NConfirmButton? confirmButton)
    {
        var singlePreview = screen.GetNodeOrNull<Control>("%UpgradeSinglePreviewContainer");
        if (singlePreview?.Visible == true)
        {
            confirmButton = singlePreview.GetNodeOrNull<NConfirmButton>("Confirm");
            return confirmButton?.IsEnabled == true;
        }

        var multiPreview = screen.GetNodeOrNull<Control>("%UpgradeMultiPreviewContainer");
        if (multiPreview?.Visible == true)
        {
            confirmButton = multiPreview.GetNodeOrNull<NConfirmButton>("Confirm");
            return confirmButton?.IsEnabled == true;
        }

        confirmButton = null;
        return false;
    }

    private static bool TryGetDeckTransformConfirmButton(
        NDeckTransformSelectScreen screen,
        out NConfirmButton? confirmButton)
    {
        var previewContainer = screen.GetNodeOrNull<Control>("%PreviewContainer");
        if (previewContainer?.Visible == true)
        {
            confirmButton = previewContainer.GetNodeOrNull<NConfirmButton>("Confirm");
            return confirmButton?.IsEnabled == true;
        }

        confirmButton = null;
        return false;
    }

    private static bool TryGetDeckEnchantConfirmButton(
        NDeckEnchantSelectScreen screen,
        out NConfirmButton? confirmButton)
    {
        var singlePreview = screen.GetNodeOrNull<Control>("%EnchantSinglePreviewContainer");
        if (singlePreview?.Visible == true)
        {
            confirmButton = singlePreview.GetNodeOrNull<NConfirmButton>("Confirm");
            return confirmButton?.IsEnabled == true;
        }

        var multiPreview = screen.GetNodeOrNull<Control>("%EnchantMultiPreviewContainer");
        if (multiPreview?.Visible == true)
        {
            confirmButton = multiPreview.GetNodeOrNull<NConfirmButton>("Confirm");
            return confirmButton?.IsEnabled == true;
        }

        confirmButton = null;
        return false;
    }

    private static async Task<bool> WaitForDeckSelectionResolutionAsync(NCardGridSelectionScreen screen, DateTime deadline)
    {
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (!GodotObject.IsInstanceValid(screen) ||
                ActiveScreenContext.Instance.GetCurrentScreen() is not NCardGridSelectionScreen)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<ActionResponsePayload> ExecuteOpenChestAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (currentScreen is not NTreasureRoom treasureRoom || !GameStateService.CanOpenChest(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "open_chest",
                screen
            });
        }

        var chestButton = treasureRoom.GetNodeOrNull<NButton>("%Chest")
            ?? throw new ApiException(503, "state_unavailable", "Chest button not found.", new
            {
                action = "open_chest",
                screen
            }, retryable: true);

        chestButton.ForceClick();
        var stable = await WaitForChestOpenTransitionAsync(TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "open_chest",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<bool> WaitForChestOpenTransitionAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (GameStateService.GetTreasureRelicCollection(currentScreen) != null)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<ActionResponsePayload> ExecuteChooseTreasureRelicAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseTreasureRelic(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_treasure_relic",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "choose_treasure_relic requires option_index.", new
            {
                action = "choose_treasure_relic"
            });
        }

        var relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
        if (relics == null || request.option_index < 0 || request.option_index >= relics.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_treasure_relic",
                option_index = request.option_index,
                relic_count = relics?.Count ?? 0
            });
        }

        RunManager.Instance.TreasureRoomRelicSynchronizer.PickRelicLocally(request.option_index.Value);
        var stable = await WaitForRelicPickTransitionAsync(TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "choose_treasure_relic",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteChooseEventOptionAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseEventOption(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_event_option",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "choose_event_option requires option_index.", new
            {
                action = "choose_event_option"
            });
        }

        var eventModel = RunManager.Instance.EventSynchronizer.GetLocalEvent()
            ?? throw new ApiException(503, "state_unavailable", "Event state is unavailable.", new
            {
                action = "choose_event_option",
                screen
            }, retryable: true);

        if (eventModel.IsFinished)
        {
            // 已结束事件只提供索引为 0 的合成继续选项。
            if (request.option_index != 0)
            {
                throw new ApiException(409, "invalid_target", "Event is finished. Only option_index 0 (proceed) is valid.", new
                {
                    action = "choose_event_option",
                    option_index = request.option_index,
                    is_finished = true
                });
            }

            await NEventRoom.Proceed();
            var stable = await WaitForEventScreenTransitionAsync(TimeSpan.FromSeconds(10));

            return new ActionResponsePayload
            {
                action = "choose_event_option",
                status = stable ? "completed" : "pending",
                stable = stable,
                message = stable ? "Event proceeded." : "Proceed queued but state is still transitioning.",
                state = GameStateService.BuildStatePayload()
            };
        }

        // 未结束事件按索引选择选项。
        var options = eventModel.CurrentOptions;
        if (request.option_index < 0 || request.option_index >= options.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_event_option",
                option_index = request.option_index,
                option_count = options.Count
            });
        }

        if (options[request.option_index.Value].IsLocked)
        {
            throw new ApiException(409, "invalid_target", "The selected event option is locked.", new
            {
                action = "choose_event_option",
                option_index = request.option_index
            });
        }

        RunManager.Instance.EventSynchronizer.ChooseLocalOption(request.option_index.Value);
        var stableOption = await WaitForEventOptionTransitionAsync(
            eventModel.Id?.Entry,
            BuildEventOptionSignature(eventModel),
            options.Count,
            TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "choose_event_option",
            status = stableOption ? "completed" : "pending",
            stable = stableOption,
            message = stableOption ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    /// <summary>
    /// 等待页面离开 NEventRoom，用于事件结束后的继续操作。
    /// </summary>
    private static async Task<bool> WaitForEventScreenTransitionAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is not NEventRoom)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// 选择事件选项后等待状态变化。页面、IsFinished 或选项数量任一变化即视为完成。
    /// </summary>
    private static async Task<bool> WaitForEventOptionTransitionAsync(
        string? previousEventId,
        string previousOptionSignature,
        int previousOptionCount,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();

            // 页面已经完全切换，例如事件触发了战斗。
            if (currentScreen is not NEventRoom)
            {
                return true;
            }

            var currentEventModel = RunManager.Instance.EventSynchronizer.GetLocalEvent();
            if (currentEventModel == null)
            {
                continue;
            }

            if (currentEventModel.Id?.Entry != previousEventId)
            {
                return true;
            }

            if (currentEventModel.IsFinished)
            {
                return true;
            }

            if (currentEventModel.CurrentOptions.Count != previousOptionCount)
            {
                return true;
            }

            if (BuildEventOptionSignature(currentEventModel) != previousOptionSignature)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<ActionResponsePayload> ExecuteChooseCapstoneOptionAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseCapstoneOption(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_capstone_option",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "choose_capstone_option requires option_index.", new
            {
                action = "choose_capstone_option"
            });
        }

        var buttons = GameStateService.GetCapstoneButtons(currentScreen);
        if (request.option_index < 0 || request.option_index >= buttons.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_capstone_option",
                option_index = request.option_index,
                button_count = buttons.Count
            });
        }

        var button = buttons[request.option_index.Value];
        button.EmitSignal(BaseButton.SignalName.Pressed);

        // 等待页面切换。
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        var stable = false;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();
            var newScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (newScreen is not NCapstoneSubmenuStack)
            {
                stable = true;
                break;
            }
        }

        return new ActionResponsePayload
        {
            action = "choose_capstone_option",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteChooseCrystalSphereCellAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseCrystalSphereCell(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_crystal_sphere_cell",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "choose_crystal_sphere_cell requires option_index (clickable cell index).", new
            {
                action = "choose_crystal_sphere_cell"
            });
        }

        var cells = GameStateService.GetClickableCrystalSphereCells(currentScreen);
        if (request.option_index < 0 || request.option_index >= cells.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_crystal_sphere_cell",
                option_index = request.option_index,
                clickable_cell_count = cells.Count
            });
        }

        var sphereScreen = (NCrystalSphereScreen)currentScreen!;
        var divinationsBefore = GameStateService.GetCrystalSphereMinigame(sphereScreen)?.DivinationCount ?? 0;

        // 发出与游戏 CrystalSphereScreenHandler 相同的 Released 信号，让点击
        // 以原生 UI 相同的方式消耗 RNG 并结算奖励。
        var cell = cells[request.option_index.Value];
        cell.EmitSignal(NClickableControl.SignalName.Released, cell);

        // 小游戏会在点击时同步减少 DivinationCount；等待计数变化，或等待页面
        // 移交到奖励/继续流程。
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
        var stable = false;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();
            var nextScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (nextScreen is not NCrystalSphereScreen nextSphereScreen)
            {
                stable = true;
                break;
            }

            var remaining = GameStateService.GetCrystalSphereMinigame(nextSphereScreen)?.DivinationCount ?? 0;
            if (remaining < divinationsBefore)
            {
                stable = true;
                break;
            }
        }

        return new ActionResponsePayload
        {
            action = "choose_crystal_sphere_cell",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteChooseBundleAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseBundle(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_bundle",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "choose_bundle requires option_index.", new
            {
                action = "choose_bundle"
            });
        }

        var bundles = GameStateService.GetBundleOptions(currentScreen);
        if (request.option_index < 0 || request.option_index >= bundles.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_bundle",
                option_index = request.option_index,
                bundle_count = bundles.Count
            });
        }

        var bundle = bundles[request.option_index.Value];
        // 直接调用页面的 OnBundleClicked 方法。
        if (currentScreen is NChooseABundleSelectionScreen bundleScreen)
        {
            ((Node)bundleScreen).Call("OnBundleClicked", bundle);
        }

        var stable = await WaitForBundlePreviewAsync(TimeSpan.FromSeconds(10));

        GameStatePayload? bundleState = null;
        try { bundleState = GameStateService.BuildStatePayload(); } catch { }

        return new ActionResponsePayload
        {
            action = "choose_bundle",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = bundleState ?? new GameStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteConfirmBundleAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanConfirmBundle(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "confirm_bundle",
                screen
            });
        }

        var buttons = GameStateService.GetBundleConfirmButtons(currentScreen);
        if (buttons.Count == 0)
        {
            throw new ApiException(409, "invalid_action", "No confirm button found.", new
            {
                action = "confirm_bundle",
                screen
            });
        }

        var confirmBtn = buttons[0];
        Log.Info($"[STS2AIAgent] confirm_bundle: clicking {confirmBtn.GetType().Name} '{confirmBtn.Name}'");

        // 优先尝试 ForceClick。
        confirmBtn.ForceClick();
        await WaitForNextFrameAsync();

        // 如果仍停留在卡包页面，尝试调用页面的 OnConfirmPressed。
        var stable = false;
        if (ActiveScreenContext.Instance.GetCurrentScreen() is NChooseABundleSelectionScreen bundleScreen2)
        {
            try
            {
                Log.Info("[STS2AIAgent] confirm_bundle: trying OnConfirmPressed");
                ((Node)bundleScreen2).Call("OnConfirmPressed");
            }
            catch { }

            // 同时尝试发送不带参数的按钮信号。
            try
            {
                confirmBtn.EmitSignal("pressed");
            }
            catch { }
        }

        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
        while (!stable && DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();
            var newScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (newScreen is not NChooseABundleSelectionScreen)
            {
                stable = true;
            }
        }

        GameStatePayload? state = null;
        try { state = GameStateService.BuildStatePayload(); } catch { }

        return new ActionResponsePayload
        {
            action = "confirm_bundle",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = state ?? new GameStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteChooseRestOptionAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanChooseRestOption(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "choose_rest_option",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "choose_rest_option requires option_index.", new
            {
                action = "choose_rest_option"
            });
        }

        var options = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
        if (options == null || request.option_index < 0 || request.option_index >= options.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_rest_option",
                option_index = request.option_index,
                option_count = options?.Count ?? 0
            });
        }

        if (!options[request.option_index.Value].IsEnabled)
        {
            throw new ApiException(409, "invalid_target", "The selected rest option is disabled.", new
            {
                action = "choose_rest_option",
                option_index = request.option_index
            });
        }

        var selectedOption = options[request.option_index.Value];
        var initialOptionsSignature = BuildRestOptionSignature(options);
        var runState = RunManager.Instance.DebugOnlyGetState();
        var localPlayer = GameStateService.GetLocalPlayer(runState);
        var requiresTarget = GameStateService.RestOptionRequiresTarget(selectedOption, runState, localPlayer);
        Player? targetPlayer = null;
        if (requiresTarget)
        {
            targetPlayer = ResolveRestOptionTarget(request, runState, localPlayer, selectedOption);
        }

        var chooseTask = RunManager.Instance.RestSiteSynchronizer.ChooseLocalOption(request.option_index.Value);

        bool stable;
        if (requiresTarget)
        {
            stable = await CompleteRestOptionTargetSelectionAsync(chooseTask, targetPlayer!, TimeSpan.FromSeconds(10));
        }
        else
        {
            stable = await WaitForRestOptionTransitionAsync(
                TimeSpan.FromSeconds(10),
                chooseTask,
                initialOptionsSignature);
            ObserveBackgroundResult(chooseTask, "choose_rest_option");
        }

        return new ActionResponsePayload
        {
            action = "choose_rest_option",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<bool> WaitForBundlePreviewAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();
            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is not NChooseABundleSelectionScreen)
            {
                return true;
            }

            if (GameStateService.GetSelectedBundle(currentScreen) != null &&
                GameStateService.CanConfirmBundle(currentScreen))
            {
                return true;
            }
        }

        return false;
    }

    private static Player ResolveRestOptionTarget(
        ActionRequest request,
        RunState? runState,
        Player? localPlayer,
        RestSiteOption selectedOption)
    {
        var targetIndexSpace = GameStateService.GetRestOptionTargetIndexSpace(selectedOption, runState, localPlayer) ?? "run.players";
        var validTargetIndices = GameStateService.GetRestOptionTargetIndices(runState, localPlayer, allowSelf: false);
        if (request.target_index == null)
        {
            throw new ApiException(409, "invalid_target", "This rest option requires target_index.", new
            {
                action = "choose_rest_option",
                option_index = request.option_index,
                option_id = selectedOption.OptionId,
                target_index_space = targetIndexSpace,
                valid_target_indices = validTargetIndices
            });
        }

        if (!validTargetIndices.Contains(request.target_index.Value))
        {
            throw new ApiException(409, "invalid_target", "target_index is out of range for run.players[].", new
            {
                action = "choose_rest_option",
                option_index = request.option_index,
                option_id = selectedOption.OptionId,
                target_index = request.target_index,
                target_index_space = targetIndexSpace,
                valid_target_indices = validTargetIndices
            });
        }

        return GameStateService.ResolveRunPlayerTarget(runState, request.target_index.Value)
            ?? throw new ApiException(409, "invalid_target", "target_index is out of range for run.players[].", new
            {
                action = "choose_rest_option",
                option_index = request.option_index,
                option_id = selectedOption.OptionId,
                target_index = request.target_index,
                target_index_space = targetIndexSpace,
                valid_target_indices = validTargetIndices
            });
    }

    private static async Task<bool> CompleteRestOptionTargetSelectionAsync(
        Task<bool> chooseTask,
        Player targetPlayer,
        TimeSpan timeout)
    {
        var targetManager = await WaitForTargetManagerSelectionAsync(timeout);
        if (targetManager == null)
        {
            ObserveBackgroundResult(chooseTask, "choose_rest_option");
            return false;
        }

        var targetNode = ResolveRestSiteTargetNode(targetPlayer)
            ?? throw new ApiException(503, "state_unavailable", "Rest-site target node is unavailable.", new
            {
                action = "choose_rest_option",
                target_player_id = targetPlayer.NetId.ToString()
            }, retryable: true);

        targetManager.OnNodeHovered(targetNode);
        targetManager.Call(NTargetManager.MethodName.FinishTargeting, false);

        var result = await WaitForTaskResultAsync(chooseTask, timeout);
        if (result != true)
        {
            if (result == null)
            {
                ObserveBackgroundResult(chooseTask, "choose_rest_option");
            }

            return false;
        }

        return await WaitForRestOptionTransitionAsync(TimeSpan.FromSeconds(2)) || result.Value;
    }

    private static async Task<NTargetManager?> WaitForTargetManagerSelectionAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            var targetManager = NTargetManager.Instance;
            if (targetManager != null && GodotObject.IsInstanceValid(targetManager) && targetManager.IsInSelection)
            {
                return targetManager;
            }

            await WaitForNextFrameAsync();
        }

        var finalTargetManager = NTargetManager.Instance;
        return finalTargetManager != null && GodotObject.IsInstanceValid(finalTargetManager) && finalTargetManager.IsInSelection
            ? finalTargetManager
            : null;
    }

    private static Node? ResolveRestSiteTargetNode(Player targetPlayer)
    {
        var restSiteRoom = NRestSiteRoom.Instance;
        if (restSiteRoom == null || !GodotObject.IsInstanceValid(restSiteRoom))
        {
            return null;
        }

        var character = restSiteRoom.GetCharacterForPlayer(targetPlayer)
            ?? restSiteRoom.Characters.FirstOrDefault(candidate => candidate.Player.NetId == targetPlayer.NetId);
        return character != null && GodotObject.IsInstanceValid(character) ? character : null;
    }

    private static async Task<bool?> WaitForTaskResultAsync(Task<bool> task, TimeSpan timeout)
    {
        var completedTask = await Task.WhenAny(task, Task.Delay(timeout));
        if (completedTask != task)
        {
            return null;
        }

        return await task;
    }

    /// <summary>
    /// 选择休息点选项后等待任务完成或状态变化。进入后续页面、出现继续按钮、
    /// 选项列表发生变化或游戏任务完成时，均视为状态已经推进。
    /// </summary>
    /// <param name="timeout">等待游戏进入可再次观测状态的最长时间。</param>
    /// <param name="chooseTask">可选的游戏选项任务；完成时与界面转换等价。</param>
    /// <param name="initialOptionsSignature">执行动作前的选项列表签名。</param>
    private static async Task<bool> WaitForRestOptionTransitionAsync(
        TimeSpan timeout,
        Task<bool>? chooseTask = null,
        string? initialOptionsSignature = null)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();

            // 页面已经完全切换，例如 SMITH 打开了选牌页。
            if (currentScreen is not NRestSiteRoom restSiteRoom)
            {
                return true;
            }

            // 继续按钮已经可用，例如 HEAL 结算完成。
            var proceedButton = restSiteRoom.ProceedButton;
            if (proceedButton != null && GodotObject.IsInstanceValid(proceedButton) && proceedButton.IsEnabled)
            {
                return true;
            }

            var options = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
            if (options != null)
            {
                var optionsChanged = initialOptionsSignature != null
                    && BuildRestOptionSignature(options) != initialOptionsSignature;
                if (options.Count == 0 || options.All(static option => !option.IsEnabled))
                {
                    restSiteRoom.Call(NRestSiteRoom.MethodName.ShowProceedButton);
                    ActiveScreenContext.Instance.Update();

                    await WaitForNextFrameAsync();
                    proceedButton = restSiteRoom.ProceedButton;
                    return optionsChanged
                        || proceedButton != null
                        && GodotObject.IsInstanceValid(proceedButton)
                        && proceedButton.IsEnabled;
                }

                if (optionsChanged)
                {
                    return true;
                }
            }

            if (chooseTask?.IsCompleted == true)
            {
                var taskResult = await chooseTask;
                if (taskResult)
                {
                    return true;
                }

                chooseTask = null;
            }
        }

        return false;
    }

    /// <summary>
    /// 生成只包含休息选项身份与可用性的稳定签名，用于识别帐篷等连续选择。
    /// </summary>
    private static string BuildRestOptionSignature(IEnumerable<RestSiteOption> options) =>
        string.Join("|", options.Select((option, index) =>
            $"{index}:{option.OptionId}:{option.IsEnabled}"));

    private static async Task<ActionResponsePayload> ExecuteOpenShopInventoryAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanOpenShopInventory(currentScreen) || currentScreen is not NMerchantRoom merchantRoom)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "open_shop_inventory",
                screen
            });
        }

        merchantRoom.OpenInventory();
        var stable = await WaitForShopInventoryOpenAsync(TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "open_shop_inventory",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteCloseShopInventoryAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanCloseShopInventory(currentScreen) || currentScreen is not NMerchantInventory inventoryScreen)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "close_shop_inventory",
                screen
            });
        }

        var backButton = inventoryScreen.GetNodeOrNull<NButton>("%BackButton")
            ?? throw new ApiException(503, "state_unavailable", "Shop back button not found.", new
            {
                action = "close_shop_inventory",
                screen
            }, retryable: true);

        backButton.ForceClick();
        var stable = await WaitForShopInventoryCloseAsync(TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "close_shop_inventory",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteBuyCardAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanBuyShopCard(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "buy_card",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "buy_card requires option_index.", new
            {
                action = "buy_card"
            });
        }

        var inventory = GameStateService.GetMerchantInventory(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Shop inventory is unavailable.", new
            {
                action = "buy_card",
                screen
            }, retryable: true);

        var cards = GameStateService.GetMerchantCardEntries(currentScreen).ToList();
        if (request.option_index < 0 || request.option_index >= cards.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "buy_card",
                option_index = request.option_index,
                option_count = cards.Count
            });
        }

        var entry = cards[request.option_index.Value];
        if (!entry.IsStocked)
        {
            throw new ApiException(409, "invalid_target", "The selected card is out of stock.", new
            {
                action = "buy_card",
                option_index = request.option_index
            });
        }

        var previousGold = inventory.Player.Gold;
        var previousCardId = entry.CreationResult?.Card.Id.Entry;
        var success = await entry.OnTryPurchaseWrapper(inventory);
        if (!success)
        {
            throw new ApiException(409, "invalid_action", "Card purchase failed in the current state.", new
            {
                action = "buy_card",
                option_index = request.option_index
            });
        }

        var stable = await WaitForMerchantCardPurchaseAsync(inventory.Player, entry, previousGold, previousCardId, TimeSpan.FromSeconds(10));
        return new ActionResponsePayload
        {
            action = "buy_card",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteBuyRelicAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanBuyShopRelic(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "buy_relic",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "buy_relic requires option_index.", new
            {
                action = "buy_relic"
            });
        }

        var inventory = GameStateService.GetMerchantInventory(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Shop inventory is unavailable.", new
            {
                action = "buy_relic",
                screen
            }, retryable: true);

        var relics = GameStateService.GetMerchantRelicEntries(currentScreen).ToList();
        if (request.option_index < 0 || request.option_index >= relics.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "buy_relic",
                option_index = request.option_index,
                option_count = relics.Count
            });
        }

        var entry = relics[request.option_index.Value];
        if (!entry.IsStocked)
        {
            throw new ApiException(409, "invalid_target", "The selected relic is out of stock.", new
            {
                action = "buy_relic",
                option_index = request.option_index
            });
        }

        var previousGold = inventory.Player.Gold;
        var previousRelicId = entry.Model?.Id.Entry;
        var success = await entry.OnTryPurchaseWrapper(inventory);
        if (!success)
        {
            throw new ApiException(409, "invalid_action", "Relic purchase failed in the current state.", new
            {
                action = "buy_relic",
                option_index = request.option_index
            });
        }

        var stable = await WaitForMerchantRelicPurchaseAsync(inventory.Player, entry, previousGold, previousRelicId, TimeSpan.FromSeconds(10));
        return new ActionResponsePayload
        {
            action = "buy_relic",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteBuyPotionAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanBuyShopPotion(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "buy_potion",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "buy_potion requires option_index.", new
            {
                action = "buy_potion"
            });
        }

        var inventory = GameStateService.GetMerchantInventory(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Shop inventory is unavailable.", new
            {
                action = "buy_potion",
                screen
            }, retryable: true);

        var potions = GameStateService.GetMerchantPotionEntries(currentScreen).ToList();
        if (request.option_index < 0 || request.option_index >= potions.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "buy_potion",
                option_index = request.option_index,
                option_count = potions.Count
            });
        }

        var entry = potions[request.option_index.Value];
        if (!entry.IsStocked)
        {
            throw new ApiException(409, "invalid_target", "The selected potion is out of stock.", new
            {
                action = "buy_potion",
                option_index = request.option_index
            });
        }

        var previousGold = inventory.Player.Gold;
        var previousPotionId = entry.Model?.Id.Entry;
        var success = await entry.OnTryPurchaseWrapper(inventory);
        if (!success)
        {
            throw new ApiException(409, "invalid_action", "Potion purchase failed in the current state.", new
            {
                action = "buy_potion",
                option_index = request.option_index
            });
        }

        var stable = await WaitForMerchantPotionPurchaseAsync(inventory.Player, entry, previousGold, previousPotionId, TimeSpan.FromSeconds(10));
        return new ActionResponsePayload
        {
            action = "buy_potion",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteRemoveCardAtShopAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanRemoveCardAtShop(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "remove_card_at_shop",
                screen
            });
        }

        var inventory = GameStateService.GetMerchantInventory(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Shop inventory is unavailable.", new
            {
                action = "remove_card_at_shop",
                screen
            }, retryable: true);

        var entry = GameStateService.GetMerchantCardRemovalEntry(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Shop card removal service is unavailable.", new
            {
                action = "remove_card_at_shop",
                screen
            }, retryable: true);

        // 商店删牌会打开牌组选择并阻塞到玩家确认，因此这里触发后即返回，
        // 不等待整个任务完成。
        ObserveBackgroundResult(entry.OnTryPurchaseWrapper(inventory), "remove_card_at_shop");
        var stable = await WaitForShopCardRemovalTransitionAsync(TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "remove_card_at_shop",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteSelectCharacterAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var multiplayerTestScene = GameStateService.GetMultiplayerTestScene();

        if (multiplayerTestScene != null)
        {
            return await ExecuteSelectMultiplayerLobbyCharacterAsync(request, multiplayerTestScene, screen);
        }

        if (currentScreen is not NCharacterSelectScreen characterSelectScreen || !GameStateService.CanSelectCharacter(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "select_character",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "select_character requires option_index.", new
            {
                action = "select_character"
            });
        }

        var buttons = GameStateService.GetCharacterSelectButtons(currentScreen);
        if (request.option_index < 0 || request.option_index >= buttons.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "select_character",
                option_index = request.option_index,
                option_count = buttons.Count
            });
        }

        var button = buttons[request.option_index.Value];
        if (button.IsLocked)
        {
            throw new ApiException(409, "invalid_target", "The selected character is locked.", new
            {
                action = "select_character",
                option_index = request.option_index,
                character_id = button.Character.Id.Entry
            });
        }

        if (!button.IsEnabled || !button.IsVisibleInTree())
        {
            throw new ApiException(409, "invalid_target", "The selected character cannot be chosen right now.", new
            {
                action = "select_character",
                option_index = request.option_index,
                character_id = button.Character.Id.Entry
            });
        }

        var previousCharacterId = characterSelectScreen.Lobby.LocalPlayer.character.Id.Entry;
        button.Select();
        var stable = await WaitForCharacterSelectionTransitionAsync(characterSelectScreen, button.Character.Id.Entry, previousCharacterId, TimeSpan.FromSeconds(5));

        return new ActionResponsePayload
        {
            action = "select_character",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteSelectMultiplayerLobbyCharacterAsync(ActionRequest request, NMultiplayerTest scene, string screen)
    {
        if (!GameStateService.CanSelectCharacter(ActiveScreenContext.Instance.GetCurrentScreen()))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "select_character",
                screen
            });
        }

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "select_character requires option_index.", new
            {
                action = "select_character"
            });
        }

        var characters = GameStateService.GetMultiplayerLobbyCharacters();
        if (request.option_index < 0 || request.option_index >= characters.Length)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "select_character",
                option_index = request.option_index,
                option_count = characters.Length
            });
        }

        var paginator = GameStateService.GetMultiplayerTestCharacterPaginator(scene)
            ?? throw new ApiException(503, "state_unavailable", "Multiplayer character selector is unavailable.", new
            {
                action = "select_character",
                screen
            }, retryable: true);

        var lobby = GameStateService.GetMultiplayerTestLobby(scene)
            ?? throw new ApiException(503, "state_unavailable", "Multiplayer lobby is unavailable.", new
            {
                action = "select_character",
                screen
            }, retryable: true);

        var previousCharacterId = lobby.LocalPlayer.character.Id.Entry;
        var currentCharacterId = characters[request.option_index.Value].Id.Entry;
        paginator.SetIndex(request.option_index.Value);
        var stable = await WaitForMultiplayerLobbyCharacterSelectionTransitionAsync(scene, currentCharacterId, previousCharacterId, TimeSpan.FromSeconds(5));

        return new ActionResponsePayload
        {
            action = "select_character",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteEmbarkAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (!GameStateService.CanEmbark(currentScreen) || currentScreen is not NCharacterSelectScreen characterSelectScreen)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "embark",
                screen
            });
        }

        var embarkButton = GameStateService.GetCharacterEmbarkButton(currentScreen)
            ?? throw new ApiException(503, "state_unavailable", "Embark button is unavailable.", new
            {
                action = "embark",
                screen
            }, retryable: true);

        embarkButton.ForceClick();
        var stable = await WaitForEmbarkTransitionAsync(characterSelectScreen, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "embark",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteUnreadyAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var multiplayerTestScene = GameStateService.GetMultiplayerTestScene();

        if (multiplayerTestScene != null)
        {
            var multiplayerLobby = GameStateService.GetMultiplayerTestLobby(multiplayerTestScene)
                ?? throw new ApiException(503, "state_unavailable", "Multiplayer lobby is unavailable.", new
                {
                    action = "unready",
                    screen
                }, retryable: true);

            if (!GameStateService.CanUnready(currentScreen))
            {
                throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
                {
                    action = "unready",
                    screen
                });
            }

            multiplayerLobby.SetReady(ready: false);
            var multiplayerStable = await WaitForMultiplayerLobbyReadyTransitionAsync(multiplayerTestScene, ready: false, expectRunStart: false, TimeSpan.FromSeconds(5));

            return new ActionResponsePayload
            {
                action = "unready",
                status = multiplayerStable ? "completed" : "pending",
                stable = multiplayerStable,
                message = multiplayerStable ? "Action completed." : "Action queued but state is still transitioning.",
                state = GameStateService.BuildStatePayload()
            };
        }

        if (!GameStateService.CanUnready(currentScreen) || currentScreen is not NCharacterSelectScreen characterSelectScreen)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "unready",
                screen
            });
        }

        characterSelectScreen.Lobby.SetReady(ready: false);
        var stable = await WaitForLobbyReadyTransitionAsync(characterSelectScreen, ready: false, TimeSpan.FromSeconds(5));

        return new ActionResponsePayload
        {
            action = "unready",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteHostMultiplayerLobbyAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var scene = GameStateService.GetMultiplayerTestScene();

        if (!GameStateService.CanHostMultiplayerLobby(currentScreen) || scene == null)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "host_multiplayer_lobby",
                screen
            });
        }

        var startHostTask = InvokePrivateTask<bool>(scene, "StartHost", false)
            ?? throw new ApiException(503, "state_unavailable", "Multiplayer host entry point is unavailable.", new
            {
                action = "host_multiplayer_lobby",
                screen
            }, retryable: true);

        var hostStarted = await startHostTask;
        if (!hostStarted)
        {
            throw new ApiException(409, "invalid_action", "Failed to host the multiplayer lobby.", new
            {
                action = "host_multiplayer_lobby",
                screen
            });
        }

        var stable = await WaitForMultiplayerLobbyHostTransitionAsync(scene, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "host_multiplayer_lobby",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteJoinMultiplayerLobbyAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var scene = GameStateService.GetMultiplayerTestScene();

        if (!GameStateService.CanJoinMultiplayerLobby(currentScreen) || scene == null)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "join_multiplayer_lobby",
                screen
            });
        }

        var joinHost = GameStateService.GetMultiplayerLobbyJoinHost();
        var joinPort = (ushort)GameStateService.GetMultiplayerLobbyJoinPort();
        var joinNetId = GameStateService.GetMultiplayerLobbyJoinNetIdHint();
        var initializer = new ENetClientConnectionInitializer(joinNetId, joinHost, joinPort);
        await scene.JoinToHost(initializer);

        if (GameStateService.GetMultiplayerTestLobby(scene) == null)
        {
            throw new ApiException(409, "invalid_action", "Failed to join the multiplayer lobby.", new
            {
                action = "join_multiplayer_lobby",
                screen,
                join_host = joinHost,
                join_port = joinPort,
                net_id = joinNetId
            });
        }

        var stable = await WaitForMultiplayerLobbyJoinTransitionAsync(scene, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "join_multiplayer_lobby",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteReadyMultiplayerLobbyAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var scene = GameStateService.GetMultiplayerTestScene();

        if (!GameStateService.CanReadyMultiplayerLobby(currentScreen) || scene == null)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "ready_multiplayer_lobby",
                screen
            });
        }

        var lobby = GameStateService.GetMultiplayerTestLobby(scene)
            ?? throw new ApiException(503, "state_unavailable", "Multiplayer lobby is unavailable.", new
            {
                action = "ready_multiplayer_lobby",
                screen
            }, retryable: true);
        var expectRunStart = lobby.Players.Count > 1 &&
            lobby.Players
                .Where(player => player.id != lobby.LocalPlayer.id)
                .All(player => player.isReady);

        InvokePrivateVoid(scene, "ReadyButtonPressed");
        var stable = await WaitForMultiplayerLobbyReadyTransitionAsync(scene, ready: true, expectRunStart, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "ready_multiplayer_lobby",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteDisconnectMultiplayerLobbyAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var scene = GameStateService.GetMultiplayerTestScene();

        if (!GameStateService.CanDisconnectMultiplayerLobby(currentScreen) || scene == null)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "disconnect_multiplayer_lobby",
                screen
            });
        }

        InvokePrivateVoid(scene, "Disconnect", NetError.Quit);
        var stable = await WaitForMultiplayerLobbyDisconnectTransitionAsync(scene, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "disconnect_multiplayer_lobby",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteAdjustAscensionAsync(int delta, string actionName)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var canAdjust = delta > 0
            ? GameStateService.CanIncreaseAscension(currentScreen)
            : GameStateService.CanDecreaseAscension(currentScreen);

        if (!canAdjust || currentScreen is not NCharacterSelectScreen characterSelectScreen)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = actionName,
                screen
            });
        }

        var targetAscension = characterSelectScreen.Lobby.Ascension + delta;
        characterSelectScreen.Lobby.SyncAscensionChange(targetAscension);
        var stable = await WaitForLobbyAscensionTransitionAsync(characterSelectScreen, targetAscension, TimeSpan.FromSeconds(5));

        return new ActionResponsePayload
        {
            action = actionName,
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteUsePotionAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var combatState = CombatManager.Instance.DebugOnlyGetState();
        var runState = RunManager.Instance.DebugOnlyGetState();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "use_potion requires option_index.", new
            {
                action = "use_potion"
            });
        }

        if (!GameStateService.CanUsePotionAtIndex(currentScreen, combatState, runState, request.option_index.Value))
        {
            throw new ApiException(409, "invalid_action", "The selected potion cannot be used in the current state.", new
            {
                action = "use_potion",
                screen,
                option_index = request.option_index
            });
        }

        var player = GameStateService.GetLocalPlayer(runState)
            ?? throw new ApiException(503, "state_unavailable", "Local player is unavailable.", new
            {
                action = "use_potion",
                screen
            }, retryable: true);

        if (request.option_index < 0 || request.option_index >= player.PotionSlots.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "use_potion",
                option_index = request.option_index,
                option_count = player.PotionSlots.Count
            });
        }

        var potion = player.PotionSlots[request.option_index.Value]
            ?? throw new ApiException(409, "invalid_target", "The selected potion slot is empty.", new
            {
                action = "use_potion",
                option_index = request.option_index
            });

        var target = ResolvePotionTarget(request, combatState, potion);
        potion.EnqueueManualUse(target);
        var stable = await WaitForPotionUseTransitionAsync(player, request.option_index.Value, potion, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "use_potion",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteDiscardPotionAsync(ActionRequest request)
    {
        var runState = RunManager.Instance.DebugOnlyGetState();
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (request.option_index == null)
        {
            throw new ApiException(400, "invalid_request", "discard_potion requires option_index.", new
            {
                action = "discard_potion"
            });
        }

        if (!GameStateService.CanDiscardPotionAtIndex(currentScreen, runState, request.option_index.Value))
        {
            throw new ApiException(409, "invalid_action", "The selected potion cannot be discarded in the current state.", new
            {
                action = "discard_potion",
                screen,
                option_index = request.option_index
            });
        }

        var player = GameStateService.GetLocalPlayer(runState)
            ?? throw new ApiException(503, "state_unavailable", "Local player is unavailable.", new
            {
                action = "discard_potion",
                screen
            }, retryable: true);

        if (request.option_index < 0 || request.option_index >= player.PotionSlots.Count)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "discard_potion",
                option_index = request.option_index,
                option_count = player.PotionSlots.Count
            });
        }

        var potion = player.PotionSlots[request.option_index.Value]
            ?? throw new ApiException(409, "invalid_target", "The selected potion slot is empty.", new
            {
                action = "discard_potion",
                option_index = request.option_index
            });

        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(
            player,
            (uint)request.option_index.Value,
            CombatManager.Instance.IsInProgress));
        var stable = await WaitForPotionDiscardTransitionAsync(player, request.option_index.Value, potion, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "discard_potion",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<ActionResponsePayload> ExecuteRunConsoleCommandAsync(ActionRequest request)
    {
        if (!AreDebugActionsEnabled())
        {
            throw new ApiException(409, "invalid_action", "run_console_command is disabled. Set STS2_ENABLE_DEBUG_ACTIONS=1 for development use.", new
            {
                action = "run_console_command"
            });
        }

        var command = request.command?.Trim();
        if (string.IsNullOrWhiteSpace(command))
        {
            throw new ApiException(400, "invalid_request", "command is required.", new
            {
                action = "run_console_command"
            });
        }

        NDevConsole console;
        try
        {
            console = NDevConsole.Instance;
        }
        catch (Exception ex)
        {
            throw new ApiException(503, "state_unavailable", $"Dev console is unavailable: {ex.Message}", new
            {
                action = "run_console_command",
                command
            }, retryable: true);
        }

        var devConsole = GetDevConsoleCore(console)
            ?? throw new ApiException(503, "state_unavailable", "Dev console backend is unavailable.", new
            {
                action = "run_console_command",
                command
            }, retryable: true);

        var runState = RunManager.Instance.DebugOnlyGetState();
        var player = GameStateService.GetLocalPlayer(runState);
        var result = devConsole.ProcessNetCommand(player, command);
        if (!result.success)
        {
            throw new ApiException(409, "invalid_action", string.IsNullOrWhiteSpace(result.msg) ? "Console command failed." : result.msg, new
            {
                action = "run_console_command",
                command
            });
        }

        if (result.task != null)
        {
            await result.task;
        }

        var stable = await WaitForConsoleCommandStabilityAsync(TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = "run_console_command",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable
                ? string.IsNullOrWhiteSpace(result.msg) ? "Console command executed." : result.msg
                : "Console command executed but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<bool> WaitForConsoleCommandStabilityAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (IsStableScreenState(ActiveScreenContext.Instance.GetCurrentScreen(), allowMapScreen: true))
            {
                return true;
            }
        }

        return IsStableScreenState(ActiveScreenContext.Instance.GetCurrentScreen(), allowMapScreen: true);
    }

    private static DevConsole? GetDevConsoleCore(NDevConsole console)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var field = typeof(NDevConsole).GetField("_devConsole", flags);
        return field?.GetValue(console) as DevConsole;
    }

    /// <summary>
    /// 判断开发动作与隐藏 checkpoint 审计是否显式启用。
    /// </summary>
    /// <returns>环境变量明确为真值时返回 <c>true</c>。</returns>
    internal static bool AreDebugActionsEnabled()
    {
        var raw = ReadEnvironmentVariable("STS2_ENABLE_DEBUG_ACTIONS");
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        raw = raw.Trim();

        return raw.Equals("1", StringComparison.OrdinalIgnoreCase) ||
               raw.Equals("true", StringComparison.OrdinalIgnoreCase) ||
               raw.Equals("yes", StringComparison.OrdinalIgnoreCase) ||
               raw.Equals("on", StringComparison.OrdinalIgnoreCase);
    }

    private static string? ReadEnvironmentVariable(string name)
    {
        var processValue = System.Environment.GetEnvironmentVariable(name);
        if (!string.IsNullOrWhiteSpace(processValue))
        {
            return processValue;
        }

        try
        {
            var godotValue = OS.GetEnvironment(name);
            if (!string.IsNullOrWhiteSpace(godotValue))
            {
                return godotValue;
            }
        }
        catch
        {
        }

        var userValue = System.Environment.GetEnvironmentVariable(name, System.EnvironmentVariableTarget.User);
        if (!string.IsNullOrWhiteSpace(userValue))
        {
            return userValue;
        }

        return System.Environment.GetEnvironmentVariable(name, System.EnvironmentVariableTarget.Machine);
    }

    private static async Task<ActionResponsePayload> ExecuteConfirmModalAsync()
    {
        return await ExecuteModalButtonAsync("confirm_modal", GameStateService.GetModalConfirmButton);
    }

    private static async Task<ActionResponsePayload> ExecuteDismissModalAsync()
    {
        return await ExecuteModalButtonAsync("dismiss_modal", GameStateService.GetModalCancelButton);
    }

    private static async Task<ActionResponsePayload> ExecuteReturnToMainMenuAsync()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);

        if (currentScreen is not NGameOverScreen gameOverScreen || !GameStateService.CanReturnToMainMenu(currentScreen))
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = "return_to_main_menu",
                screen
            });
        }

        gameOverScreen.Call(NGameOverScreen.MethodName.ReturnToMainMenu);
        var stable = await WaitForGameOverExitAsync(TimeSpan.FromSeconds(15));

        return new ActionResponsePayload
        {
            action = "return_to_main_menu",
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<bool> WaitForShopInventoryOpenAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is NMerchantInventory inventory && inventory.IsOpen)
            {
                return true;
            }
        }

        return ActiveScreenContext.Instance.GetCurrentScreen() is NMerchantInventory openInventory && openInventory.IsOpen;
    }

    private static async Task<bool> WaitForShopInventoryCloseAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is not NMerchantInventory)
            {
                return true;
            }
        }

        return ActiveScreenContext.Instance.GetCurrentScreen() is not NMerchantInventory;
    }

    private static async Task<bool> WaitForMerchantCardPurchaseAsync(
        Player player,
        MerchantCardEntry entry,
        int previousGold,
        string? previousCardId,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentGold = player.Gold;
            var currentCardId = entry.CreationResult?.Card.Id.Entry;
            if (currentGold != previousGold || currentCardId != previousCardId || !entry.IsStocked)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<bool> WaitForMerchantRelicPurchaseAsync(
        Player player,
        MerchantRelicEntry entry,
        int previousGold,
        string? previousRelicId,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentGold = player.Gold;
            var currentRelicId = entry.Model?.Id.Entry;
            if (currentGold != previousGold || currentRelicId != previousRelicId || !entry.IsStocked)
            {
                return true;
            }
        }

        return player.Gold != previousGold || entry.Model?.Id.Entry != previousRelicId || !entry.IsStocked;
    }

    private static async Task<bool> WaitForMerchantPotionPurchaseAsync(
        Player player,
        MerchantPotionEntry entry,
        int previousGold,
        string? previousPotionId,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentGold = player.Gold;
            var currentPotionId = entry.Model?.Id.Entry;
            if (currentGold != previousGold || currentPotionId != previousPotionId || !entry.IsStocked)
            {
                return true;
            }
        }

        return player.Gold != previousGold || entry.Model?.Id.Entry != previousPotionId || !entry.IsStocked;
    }

    private static async Task<bool> WaitForShopCardRemovalTransitionAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is NCardGridSelectionScreen || currentScreen is not NMerchantInventory)
            {
                return true;
            }
        }

        var finalScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        return finalScreen is NCardGridSelectionScreen || finalScreen is not NMerchantInventory;
    }

    private static Creature? ResolvePotionTarget(ActionRequest request, CombatState? combatState, PotionModel potion)
    {
        return potion.TargetType switch
        {
            TargetType.AllEnemies => null,
            TargetType.AnyEnemy => ResolvePotionEnemyTarget(request, combatState, potion),
            TargetType.AnyPlayer when GameStateService.PotionRequiresTarget(combatState, potion) => ResolvePotionPlayerTarget(request, combatState, potion),
            TargetType.TargetedNoCreature => null,
            _ => potion.Owner.Creature
        };
    }

    private static Creature ResolvePotionEnemyTarget(ActionRequest request, CombatState? combatState, PotionModel potion)
    {
        if (combatState == null)
        {
            throw new ApiException(503, "state_unavailable", "Combat state is unavailable.", new
            {
                action = "use_potion",
                potion_id = potion.Id.Entry
            }, retryable: true);
        }

        if (request.target_index == null)
        {
            throw new ApiException(409, "invalid_target", "This potion requires target_index.", new
            {
                action = "use_potion",
                potion_id = potion.Id.Entry,
                target_type = potion.TargetType.ToString(),
                target_index_space = "enemies"
            });
        }

        var enemy = GameStateService.ResolveEnemyTarget(combatState, request.target_index.Value);
        if (enemy == null)
        {
            throw new ApiException(409, "invalid_target", "target_index is out of range for combat.enemies[].", new
            {
                action = "use_potion",
                potion_id = potion.Id.Entry,
                target_index = request.target_index,
                target_index_space = "enemies"
            });
        }

        return enemy;
    }

    private static Creature ResolvePotionPlayerTarget(ActionRequest request, CombatState? combatState, PotionModel potion)
    {
        if (combatState == null)
        {
            throw new ApiException(503, "state_unavailable", "Combat state is unavailable.", new
            {
                action = "use_potion",
                potion_id = potion.Id.Entry
            }, retryable: true);
        }

        if (request.target_index == null)
        {
            throw new ApiException(409, "invalid_target", "This potion requires target_index.", new
            {
                action = "use_potion",
                potion_id = potion.Id.Entry,
                target_type = potion.TargetType.ToString(),
                target_index_space = "players"
            });
        }

        var playerTargetIndices = GameStateService.GetTargetablePlayerIndices(combatState, potion.Owner, allowSelf: true);
        if (!playerTargetIndices.Contains(request.target_index.Value))
        {
            throw new ApiException(409, "invalid_target", "target_index is out of range for combat.players[].", new
            {
                action = "use_potion",
                potion_id = potion.Id.Entry,
                target_index = request.target_index,
                target_index_space = "players"
            });
        }

        return GameStateService.ResolvePlayerTarget(combatState, request.target_index.Value)
            ?? throw new ApiException(409, "invalid_target", "target_index is out of range for combat.players[].", new
            {
                action = "use_potion",
                potion_id = potion.Id.Entry,
                target_index = request.target_index,
                target_index_space = "players"
            });
    }

    private static NEpochSlot ResolveTimelineSlot(IScreenContext? currentScreen, int optionIndex)
    {
        var slots = GameStateService.GetTimelineSlots(currentScreen)
            .Where(slot => slot.State is EpochSlotState.Obtained or EpochSlotState.Complete)
            .ToArray();

        if (optionIndex < 0 || optionIndex >= slots.Length)
        {
            throw new ApiException(409, "invalid_target", "option_index is out of range.", new
            {
                action = "choose_timeline_epoch",
                option_index = optionIndex
            });
        }

        return slots[optionIndex];
    }

    private static async Task<bool> WaitForCharacterSelectOpenAsync(NMainMenu screen, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is NCharacterSelectScreen)
            {
                return true;
            }

            if (!GodotObject.IsInstanceValid(screen))
            {
                return true;
            }
        }

        var finalScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        return finalScreen is NCharacterSelectScreen;
    }

    private static async Task<bool> WaitForTimelineEpochTransitionAsync(
        NEpochSlot slot,
        EpochSlotState previousState,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is not NTimelineScreen)
            {
                return true;
            }

            if (GameStateService.CanConfirmTimelineOverlay(currentScreen))
            {
                return true;
            }

            if (GameStateService.GetTimelineInspectScreen(currentScreen) != null ||
                GameStateService.GetTimelineUnlockScreen(currentScreen) != null)
            {
                continue;
            }

            if (!GodotObject.IsInstanceValid(slot) || slot.State != previousState)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<bool> WaitForMainMenuSubmenuOpenAsync<TSubmenu>(NMainMenu screen, TimeSpan timeout)
        where TSubmenu : NSubmenu
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is TSubmenu)
            {
                return true;
            }

            if (!GodotObject.IsInstanceValid(screen))
            {
                return true;
            }
        }

        return ActiveScreenContext.Instance.GetCurrentScreen() is TSubmenu;
    }

    private static async Task<bool> WaitForMainMenuExitAsync(NMainMenu screen, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (GameStateService.GetOpenModal() != null)
            {
                return true;
            }

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!ReferenceEquals(currentScreen, screen) &&
                GameStateService.ResolveScreen(currentScreen) != "UNKNOWN")
            {
                return true;
            }
        }

        var finalScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        return !ReferenceEquals(finalScreen, screen) &&
               GameStateService.ResolveScreen(finalScreen) != "UNKNOWN";
    }

    private static async Task<bool> WaitForMainMenuModalAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (GameStateService.GetOpenModal() != null)
            {
                return true;
            }
        }

        return GameStateService.GetOpenModal() != null;
    }

    private static async Task<bool> WaitForTimelineInspectCloseAsync(
        NEpochInspectScreen? inspectScreen,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is not NTimelineScreen)
            {
                return true;
            }

            var currentInspect = GameStateService.GetTimelineInspectScreen(currentScreen);
            if (currentInspect == null || (inspectScreen != null && !ReferenceEquals(currentInspect, inspectScreen)))
            {
                return true;
            }

            if (GameStateService.GetTimelineUnlockScreen(currentScreen) != null)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<bool> WaitForTimelineUnlockTransitionAsync(Type unlockScreenType, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is not NTimelineScreen)
            {
                return true;
            }

            var unlockScreen = GameStateService.GetTimelineUnlockScreen(currentScreen);
            if (unlockScreen == null || unlockScreen.GetType() != unlockScreenType)
            {
                return true;
            }
        }

        return false;
    }

    private static async Task<bool> WaitForMainMenuSubmenuCloseAsync(
        NMainMenuSubmenuStack submenuStack,
        NSubmenu submenu,
        TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!ReferenceEquals(currentScreen, submenu) || !submenuStack.SubmenusOpen)
            {
                return true;
            }
        }

        var finalScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        return !ReferenceEquals(finalScreen, submenu) || !submenuStack.SubmenusOpen;
    }

    private static async Task<bool> WaitForCharacterSelectionTransitionAsync(
        NCharacterSelectScreen screen,
        string currentCharacterId,
        string previousCharacterId,
        TimeSpan timeout)
    {
        if (currentCharacterId == previousCharacterId)
        {
            return true;
        }

        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (!GodotObject.IsInstanceValid(screen))
            {
                return true;
            }

            if (screen.Lobby.LocalPlayer.character.Id.Entry == currentCharacterId)
            {
                return true;
            }
        }

        return screen.Lobby.LocalPlayer.character.Id.Entry == currentCharacterId;
    }

    private static async Task<bool> WaitForEmbarkTransitionAsync(NCharacterSelectScreen screen, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (GameStateService.GetOpenModal() != null)
            {
                return true;
            }

            if (screen.Lobby.NetService.Type.IsMultiplayer() && screen.Lobby.LocalPlayer.isReady)
            {
                return true;
            }

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (!ReferenceEquals(currentScreen, screen) &&
                GameStateService.ResolveScreen(currentScreen) != "UNKNOWN")
            {
                return true;
            }
        }

        var finalScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        if (screen.Lobby.NetService.Type.IsMultiplayer() && screen.Lobby.LocalPlayer.isReady)
        {
            return true;
        }

        return !ReferenceEquals(finalScreen, screen) &&
               GameStateService.ResolveScreen(finalScreen) != "UNKNOWN";
    }

    private static async Task<bool> WaitForLobbyReadyTransitionAsync(NCharacterSelectScreen screen, bool ready, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (!GodotObject.IsInstanceValid(screen))
            {
                return ready;
            }

            if (screen.Lobby.LocalPlayer.isReady == ready)
            {
                return true;
            }
        }

        return GodotObject.IsInstanceValid(screen) && screen.Lobby.LocalPlayer.isReady == ready;
    }

    private static async Task<bool> WaitForLobbyAscensionTransitionAsync(NCharacterSelectScreen screen, int targetAscension, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (!GodotObject.IsInstanceValid(screen))
            {
                return false;
            }

            if (screen.Lobby.Ascension == targetAscension)
            {
                return true;
            }
        }

        return GodotObject.IsInstanceValid(screen) && screen.Lobby.Ascension == targetAscension;
    }

    private static async Task<bool> WaitForMultiplayerLobbyCharacterSelectionTransitionAsync(
        NMultiplayerTest scene,
        string currentCharacterId,
        string previousCharacterId,
        TimeSpan timeout)
    {
        if (currentCharacterId == previousCharacterId)
        {
            return true;
        }

        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScene = GameStateService.GetMultiplayerTestScene();
            if (!ReferenceEquals(currentScene, scene))
            {
                return false;
            }

            var lobby = GameStateService.GetMultiplayerTestLobby(scene);
            if (lobby?.LocalPlayer.character?.Id.Entry == currentCharacterId)
            {
                return true;
            }
        }

        return GameStateService.GetMultiplayerTestLobby(scene)?.LocalPlayer.character?.Id.Entry == currentCharacterId;
    }

    private static async Task<bool> WaitForMultiplayerLobbyHostTransitionAsync(NMultiplayerTest scene, TimeSpan timeout)
    {
        return await WaitForMultiplayerLobbyTransitionAsync(scene, timeout, lobby =>
            lobby != null &&
            lobby.NetService.Type == NetGameType.Host &&
            lobby.Players.Count >= 1);
    }

    private static async Task<bool> WaitForMultiplayerLobbyJoinTransitionAsync(NMultiplayerTest scene, TimeSpan timeout)
    {
        return await WaitForMultiplayerLobbyTransitionAsync(scene, timeout, lobby =>
            lobby != null &&
            lobby.NetService.Type == NetGameType.Client &&
            lobby.Players.Count >= 2);
    }

    private static async Task<bool> WaitForMultiplayerLobbyTransitionAsync(
        NMultiplayerTest scene,
        TimeSpan timeout,
        Func<StartRunLobby?, bool> predicate)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScene = GameStateService.GetMultiplayerTestScene();
            if (!ReferenceEquals(currentScene, scene))
            {
                return false;
            }

            var lobby = GameStateService.GetMultiplayerTestLobby(scene);
            if (predicate(lobby))
            {
                return true;
            }
        }

        return predicate(GameStateService.GetMultiplayerTestLobby(scene));
    }

    private static async Task<bool> WaitForMultiplayerLobbyReadyTransitionAsync(NMultiplayerTest scene, bool ready, bool expectRunStart, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScene = GameStateService.GetMultiplayerTestScene();
            if (!ReferenceEquals(currentScene, scene))
            {
                return ready && expectRunStart;
            }

            var lobby = GameStateService.GetMultiplayerTestLobby(scene);
            if (ready && expectRunStart && lobby != null && lobby.LocalPlayer.isReady)
            {
                continue;
            }

            if (lobby != null && lobby.LocalPlayer.isReady == ready)
            {
                return true;
            }
        }

        var finalScene = GameStateService.GetMultiplayerTestScene();
        if (!ReferenceEquals(finalScene, scene))
        {
            return ready && expectRunStart;
        }

        return GameStateService.GetMultiplayerTestLobby(scene)?.LocalPlayer.isReady == ready;
    }

    private static async Task<bool> WaitForMultiplayerLobbyDisconnectTransitionAsync(NMultiplayerTest scene, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScene = GameStateService.GetMultiplayerTestScene();
            if (currentScene == null)
            {
                return true;
            }

            if (ReferenceEquals(currentScene, scene) && GameStateService.GetMultiplayerTestLobby(scene) == null)
            {
                return true;
            }
        }

        var finalScene = GameStateService.GetMultiplayerTestScene();
        return finalScene == null || (ReferenceEquals(finalScene, scene) && GameStateService.GetMultiplayerTestLobby(scene) == null);
    }

    private static async Task<bool> WaitForPotionUseTransitionAsync(Player player, int potionIndex, PotionModel potion, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (IsPotionUseAwaitingPlayerInput())
            {
                return false;
            }

            if (HasPotionUseSettled(player, potionIndex, potion))
            {
                return true;
            }
        }

        return HasPotionUseSettled(player, potionIndex, potion);
    }

    private static async Task<bool> WaitForPotionDiscardTransitionAsync(Player player, int potionIndex, PotionModel potion, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (potion.HasBeenRemovedFromState)
            {
                return true;
            }

            if (potionIndex >= player.PotionSlots.Count)
            {
                return true;
            }

            if (!ReferenceEquals(player.PotionSlots[potionIndex], potion))
            {
                return true;
            }
        }

        return potion.HasBeenRemovedFromState || !ReferenceEquals(player.PotionSlots[potionIndex], potion);
    }

    private static bool HasPotionUseSettled(Player player, int potionIndex, PotionModel potion)
    {
        if (!HasPotionSlotTransitioned(player, potionIndex, potion))
        {
            return false;
        }

        return ArePlayerDrivenActionsSettled();
    }

    private static bool HasPotionSlotTransitioned(Player player, int potionIndex, PotionModel potion)
    {
        if (potion.HasBeenRemovedFromState)
        {
            return true;
        }

        if (potionIndex >= player.PotionSlots.Count)
        {
            return true;
        }

        return !ReferenceEquals(player.PotionSlots[potionIndex], potion);
    }

    private static bool IsPotionUseAwaitingPlayerInput()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        if (currentScreen is NCardGridSelectionScreen or NChooseACardSelectionScreen)
        {
            return true;
        }

        return GameStateService.TryGetCombatHandSelection(currentScreen, out _);
    }

    private static async Task<ActionResponsePayload> ExecuteModalButtonAsync(
        string actionName,
        Func<IScreenContext?, NButton?> buttonResolver)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        var previousModal = GameStateService.GetOpenModal();
        var button = buttonResolver(currentScreen);

        if (previousModal == null || button == null)
        {
            throw new ApiException(409, "invalid_action", "Action is not available in the current state.", new
            {
                action = actionName,
                screen
            });
        }

        button.ForceClick();
        var stable = await WaitForModalTransitionAsync(previousModal, TimeSpan.FromSeconds(10));

        return new ActionResponsePayload
        {
            action = actionName,
            status = stable ? "completed" : "pending",
            stable = stable,
            message = stable ? "Action completed." : "Action queued but state is still transitioning.",
            state = GameStateService.BuildStatePayload()
        };
    }

    private static async Task<bool> WaitForModalTransitionAsync(IScreenContext previousModal, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentModal = GameStateService.GetOpenModal();
            if (currentModal == null || !ReferenceEquals(currentModal, previousModal))
            {
                return true;
            }
        }

        var finalModal = GameStateService.GetOpenModal();
        return finalModal == null || !ReferenceEquals(finalModal, previousModal);
    }

    private static async Task<bool> WaitForGameOverExitAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            if (ActiveScreenContext.Instance.GetCurrentScreen() is not NGameOverScreen)
            {
                return true;
            }
        }

        return ActiveScreenContext.Instance.GetCurrentScreen() is not NGameOverScreen;
    }

    private static Task<ActionResponsePayload> ExecuteSetSeedAsync(ActionRequest request)
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var screen = GameStateService.ResolveScreen(currentScreen);
        if (!GameStateService.CanSetSeed(currentScreen) || currentScreen is not NCharacterSelectScreen characterSelectScreen)
        {
            throw new ApiException(409, "invalid_action", "set_seed is not available in the current state.", new
            {
                action = "set_seed",
                screen
            });
        }

        var input = request.game_seed;
        var requested = StandardModeSeedService.CanonicalizeAndValidate(input);
        var resolved = StandardModeSeedService.SetAndReadBack(characterSelectScreen.Lobby, requested);
        return Task.FromResult(new ActionResponsePayload
        {
            action = "set_seed",
            status = "completed",
            stable = true,
            message = "Action completed.",
            input_game_seed = input,
            requested_game_seed = requested,
            resolved_game_seed = resolved,
            state = GameStateService.BuildStatePayload()
        });
    }

    private static async Task<NPauseMenu?> WaitForPauseMenuAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var pauseMenu = FindPauseMenu();
            if (pauseMenu != null && pauseMenu.IsVisibleInTree())
            {
                return pauseMenu;
            }
        }

        return FindPauseMenu();
    }

    private static async Task<bool> WaitForMainMenuAfterSaveAndQuitAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is NMainMenu)
            {
                return true;
            }
        }

        return ActiveScreenContext.Instance.GetCurrentScreen() is NMainMenu;
    }

    private static NPauseMenu? FindPauseMenu()
    {
        return FindFirstInGame<NPauseMenu>();
    }

    private static T? FindFirstInGame<T>() where T : Node
    {
        var game = NGame.Instance;
        if (game == null || !GodotObject.IsInstanceValid(game))
        {
            return null;
        }

        Node? root = game.GetTree()?.Root;
        root ??= game;
        return FindFirstDescendant<T>(root);
    }

    private static T? FindFirstDescendant<T>(Node? node) where T : Node
    {
        if (node == null || !GodotObject.IsInstanceValid(node))
        {
            return null;
        }

        if (node is T typedNode)
        {
            return typedNode;
        }

        foreach (var child in node.GetChildren())
        {
            var found = FindFirstDescendant<T>(child);
            if (found != null)
            {
                return found;
            }
        }

        return null;
    }

    private static string BuildEventOptionSignature(EventModel eventModel)
    {
        return string.Join(
            "|",
            eventModel.CurrentOptions.Select(option =>
                $"{option.TextKey}:{option.IsLocked}:{option.IsProceed}:{option.Title?.GetFormattedText()}:{option.Description?.GetFormattedText()}"));
    }

    private static void ObserveBackgroundResult(Task<bool> task, string actionName)
    {
        _ = ObserveBackgroundResultCore(task, actionName);
    }

    private static Task<T>? InvokePrivateTask<T>(object target, string methodName, params object?[] args)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var method = target.GetType().GetMethod(methodName, flags);
        return method?.Invoke(target, args) as Task<T>;
    }

    private static Task? InvokePrivateTask(object target, string methodName, params object?[] args)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var method = target.GetType().GetMethod(methodName, flags);
        return method?.Invoke(target, args) as Task;
    }

    private static T? GetPrivateField<T>(object target, string fieldName) where T : class
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var field = target.GetType().GetField(fieldName, flags);
        return field?.GetValue(target) as T;
    }

    private static void InvokePrivateVoid(object target, string methodName, params object?[] args)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var method = target.GetType().GetMethod(methodName, flags)
            ?? throw new InvalidOperationException($"Method '{methodName}' was not found on {target.GetType().FullName}.");
        method.Invoke(target, args);
    }

    private static async Task ObserveBackgroundResultCore(Task<bool> task, string actionName)
    {
        try
        {
            var success = await task;
            if (!success)
            {
                Log.Warn($"[STS2AIAgent] Background action {actionName} returned false.");
            }
        }
        catch (Exception ex)
        {
            Log.Error($"[STS2AIAgent] Background action {actionName} failed: {ex}");
        }
    }

    private static async Task<bool> WaitForRelicPickTransitionAsync(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            await WaitForNextFrameAsync();

            var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
            if (currentScreen is NTreasureRoomRelicCollection)
            {
                continue;
            }

            if (currentScreen is NTreasureRoom)
            {
                if (GameStateService.GetProceedButton(currentScreen) != null)
                {
                    await WaitForNextFrameAsync();

                    var confirmedScreen = ActiveScreenContext.Instance.GetCurrentScreen();
                    return confirmedScreen is NTreasureRoom && GameStateService.GetProceedButton(confirmedScreen) != null;
                }

                continue;
            }

            if (IsStableScreenState(currentScreen, allowMapScreen: true))
            {
                return true;
            }
        }

        var screen = ActiveScreenContext.Instance.GetCurrentScreen();
        return screen is NTreasureRoom && GameStateService.GetProceedButton(screen) != null;
    }

    /// <summary>
    /// 通过 Godot 的 ProcessFrame 信号等待下一游戏帧。NGame 或 SceneTree
    /// 不可用时（例如关闭期间），回退到不带 ConfigureAwait(false) 的
    /// Task.Delay，以保留游戏线程的 SynchronizationContext。这里不能使用
    /// ConfigureAwait(false)，否则后续循环会在线程池中执行，破坏 Godot
    /// 对象的线程访问安全。
    /// </summary>
    private static async Task WaitForNextFrameAsync()
    {
        var game = NGame.Instance;
        if (game == null || !GodotObject.IsInstanceValid(game))
        {
            await Task.Delay(TimeSpan.FromMilliseconds(16));
            return;
        }

        var tree = game.GetTree();
        if (tree == null || !GodotObject.IsInstanceValid(tree))
        {
            await Task.Delay(TimeSpan.FromMilliseconds(16));
            return;
        }

        await game.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
    }
}

internal sealed class ActionRequest
{
    public string? action { get; init; }

    public long? expected_state_revision { get; init; }

    public int? card_index { get; init; }

    public int? target_index { get; init; }

    public int? option_index { get; init; }

    public string? command { get; init; }

    public string? game_seed { get; init; }

    public object? client_context { get; init; }
}

internal sealed class ActionResponsePayload
{
    public string action { get; init; } = string.Empty;

    public string status { get; init; } = "failed";

    public bool stable { get; init; }

    public string message { get; init; } = string.Empty;

    public string? input_game_seed { get; init; }

    public string? requested_game_seed { get; init; }

    public string? resolved_game_seed { get; init; }

    public GameStatePayload state { get; init; } = new();
}
