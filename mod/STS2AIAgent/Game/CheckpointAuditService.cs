using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using STS2AIAgent.Server;

namespace STS2AIAgent.Game;

/// <summary>
/// 导出只供 checkpoint 确定性测试使用的隐藏环境状态。
/// </summary>
internal static class CheckpointAuditService
{
    /// <summary>
    /// 读取当前整局 RNG、房间、玩家随机状态与完整战斗牌堆。
    /// </summary>
    /// <returns>不会进入普通 <c>/state</c> 的隐藏审计对象。</returns>
    /// <exception cref="ApiException">开发审计未启用或当前没有进行中的整局。</exception>
    internal static object Build()
    {
        if (!GameActionService.AreDebugActionsEnabled())
        {
            throw new ApiException(
                409,
                "checkpoint_audit_disabled",
                "Checkpoint audit requires STS2_ENABLE_DEBUG_ACTIONS=1.");
        }

        var runState = RunManager.Instance.DebugOnlyGetState();
        if (runState == null)
        {
            throw new ApiException(
                409,
                "run_unavailable",
                "Checkpoint audit requires an active run.");
        }

        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        var combatState = CombatManager.Instance.DebugOnlyGetState();
        var combatActive = CombatManager.Instance.IsInProgress;
        return new
        {
            screen = GameStateService.ResolveScreen(currentScreen),
            run_id = runState.Rng.StringSeed,
            run = new
            {
                current_act_index = runState.CurrentActIndex,
                act_floor = runState.ActFloor,
                total_floor = runState.TotalFloor,
                current_coord = BuildMapCoord(runState.CurrentMapCoord),
                visited_coords = runState.VisitedMapCoords
                    .Select(coord => BuildMapCoord(coord))
                    .ToArray(),
                current_room = BuildRoom(runState),
                visited_event_ids = runState.VisitedEventIds
                    .Select(id => id.Entry)
                    .OrderBy(id => id, StringComparer.Ordinal)
                    .ToArray(),
                odds = JsonHelper.Serialize(runState.Odds.ToSerializable()),
                shared_relic_grab_bag = JsonHelper.Serialize(
                    runState.SharedRelicGrabBag.ToSerializable())
            },
            run_rng = BuildRunRng(runState),
            players = runState.Players
                .OrderBy(runState.GetPlayerSlotIndex)
                .Select(player => BuildPlayer(runState, player, combatActive))
                .ToArray(),
            combat = combatActive
                ? BuildCombat(combatState)
                : null
        };
    }

    /// <summary>
    /// 导出全部十二条整局 RNG 流的内部状态。
    /// </summary>
    /// <param name="runState">当前整局状态。</param>
    /// <returns>按游戏枚举名称索引的 RNG 状态。</returns>
    private static Dictionary<string, object> BuildRunRng(RunState runState)
    {
        return Enum.GetValues<RunRngType>().ToDictionary(
            type => type.ToString(),
            type => BuildRng(runState.Rng.GetRng(type)),
            StringComparer.Ordinal);
    }

    /// <summary>
    /// 导出一名玩家的 RNG、概率袋、牌组与全部战斗牌堆。
    /// </summary>
    /// <param name="runState">当前整局状态。</param>
    /// <param name="player">待审计玩家。</param>
    /// <param name="combatActive">当前是否存在仍在进行的战斗。</param>
    /// <returns>不含进程地址或墙钟时间的玩家审计对象。</returns>
    private static object BuildPlayer(
        RunState runState,
        Player player,
        bool combatActive)
    {
        IEnumerable<CardPile> piles = combatActive
            ? player.Piles
            : new[] { player.Deck };
        return new
        {
            player_id = player.NetId.ToString(),
            slot_index = runState.GetPlayerSlotIndex(player),
            character_id = player.Character.Id.Entry,
            current_hp = player.Creature.CurrentHp,
            max_hp = player.Creature.MaxHp,
            gold = player.Gold,
            player_rng = Enum.GetValues<PlayerRngType>().ToDictionary(
                type => type.ToString(),
                type => BuildRng(player.PlayerRng.GetRng(type)),
                StringComparer.Ordinal),
            odds = JsonHelper.Serialize(player.PlayerOdds.ToSerializable()),
            relic_grab_bag = JsonHelper.Serialize(player.RelicGrabBag.ToSerializable()),
            piles = piles.Select(BuildPile).ToArray()
        };
    }

    /// <summary>
    /// 导出战斗入口的回合、遭遇、敌人状态和敌人私有 RNG。
    /// </summary>
    /// <param name="combatState">当前可空战斗状态。</param>
    /// <returns>完整战斗审计；不在战斗中时为空。</returns>
    private static object? BuildCombat(CombatState? combatState)
    {
        if (combatState == null)
        {
            return null;
        }

        return new
        {
            encounter_id = combatState.Encounter?.Id.Entry,
            round = combatState.RoundNumber,
            current_side = combatState.CurrentSide.ToString(),
            enemies = combatState.Enemies.Select((enemy, index) => new
            {
                index,
                enemy_id = enemy.ModelId.Entry,
                current_hp = enemy.CurrentHp,
                max_hp = enemy.MaxHp,
                block = enemy.Block,
                is_alive = enemy.IsAlive,
                move_id = enemy.Monster?.NextMove?.Id,
                rng = enemy.Monster == null ? null : BuildRng(enemy.Monster.Rng)
            }).ToArray()
        };
    }

    /// <summary>
    /// 按当前顺序导出一个牌堆中的全部卡牌实例。
    /// </summary>
    /// <param name="pile">待审计牌堆。</param>
    /// <returns>牌堆类型和有序卡牌实例。</returns>
    private static object BuildPile(CardPile pile)
    {
        return new
        {
            pile_type = pile.Type.ToString(),
            cards = pile.Cards.Select((card, index) => new
            {
                index,
                card_id = card.Id.Entry,
                upgraded = card.IsUpgraded,
                upgrade_level = card.CurrentUpgradeLevel,
                enchantment_id = card.Enchantment?.Id.Entry,
                enchantment_amount = card.Enchantment?.Amount
            }).ToArray()
        };
    }

    /// <summary>
    /// 导出一条 RNG 的 counter 与四个内部状态字。
    /// </summary>
    /// <param name="rng">待审计随机流。</param>
    /// <returns>可逐字段比较的原生 RNG 状态。</returns>
    private static object BuildRng(Rng rng)
    {
        var state = rng.ToSerializable();
        return new
        {
            counter = state.counter,
            s0 = state.state0,
            s1 = state.state1,
            s2 = state.state2,
            s3 = state.state3
        };
    }

    /// <summary>
    /// 把游戏地图坐标转换为稳定的行列对象。
    /// </summary>
    /// <param name="coord">待导出的可空坐标。</param>
    /// <returns>行列对象；没有当前坐标时为空。</returns>
    private static object? BuildMapCoord(MapCoord? coord)
    {
        return coord.HasValue
            ? new { row = coord.Value.row, col = coord.Value.col }
            : null;
    }

    /// <summary>
    /// 导出当前房间的可序列化身份。
    /// </summary>
    /// <param name="runState">当前整局状态。</param>
    /// <returns>房间类型、模型与原生房间编号。</returns>
    private static object? BuildRoom(RunState runState)
    {
        var room = runState.CurrentRoom;
        return room == null
            ? null
            : new
            {
                room_type = room.RoomType.ToString(),
                model_id = room.ModelId?.Entry,
                is_pre_finished = room.IsPreFinished
            };
    }
}
