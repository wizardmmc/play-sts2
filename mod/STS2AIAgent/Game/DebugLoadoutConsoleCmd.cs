using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.DevConsole;
using MegaCrit.Sts2.Core.DevConsole.ConsoleCommands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Runs;

namespace STS2AIAgent.Game;

/// <summary>
/// 仅用于开发的场景命令，以原子方式布置战斗入口状态：
///   loadout cards=ZAP+1,COOLHEADEDx2 [relics=ORICHALCUM:m,...] [relics_obtain=ID,...]
///          [potions=ID,...] [potion_slots=5] [hp=65/82]
/// 卡牌后缀 "+N" 表示创建时已经升级 N 次；超过 MaxUpgradeLevel 会被拒绝。
/// 遗物后缀 ":m" 表示以玩具箱蜡制副本的形式授予，并处于融化耗尽状态。
/// "relics=" 使用存档读写同款的底层操作静默替换遗物，不触发拾取效果，
/// 因而不会触发 WAR_PAINT 随机升级、TOY_BOX 生成蜡制遗物、融化遗物复活，
/// 也不会改动遗物随机袋。"relics_obtain=" 保留 RelicCmd.Obtain 的拾取语义，
/// 用于测试拾取效果，并与 "relics=" 互斥。
/// "potion_slots=" 设置药水栏容量；静默授予不会触发 POTION_BELT 的容量加成。
/// "hp=" 直接设置战斗内外的当前生命值和可选最大生命值。应用任何改动之前，
/// 所有标识和值都会通过 ModelDb 和当前玩家状态完成校验，避免错误输入留下
/// 只应用一半的局面。该命令只用于构建可控的测试和训练入口状态。
/// </summary>
public sealed class DebugLoadoutConsoleCmd : AbstractConsoleCmd
{
    private static readonly Regex CardToken = new(
        @"^(?<id>[A-Z0-9_]+?)(?:\+(?<up>[1-9][0-9]*))?(?:x(?<count>[1-9][0-9]*))?$",
        RegexOptions.Compiled);
    private static readonly Regex RelicToken = new(@"^(?<id>[A-Z0-9_]+?)(?::m)?$", RegexOptions.Compiled);
    private static readonly Regex HpToken = new(@"^(?<cur>[1-9][0-9]*)(?:/(?<max>[1-9][0-9]*))?$", RegexOptions.Compiled);

    private const int MaxPotionSlots = 10;

    private sealed record CardSpec(CardModel Template, int UpgradeLevel, int Count);
    private sealed record RelicSpec(RelicModel Template, bool Melted);
    private sealed record HpSpec(int Current, int? Max);

    private sealed record LoadoutSpec(
        IReadOnlyList<CardSpec> Cards,
        IReadOnlyList<RelicSpec>? Relics,
        IReadOnlyList<RelicModel>? RelicsObtain,
        IReadOnlyList<PotionModel> Potions,
        int? PotionSlots,
        HpSpec? Hp);

    public override string CmdName => "loadout";

    public override string Args =>
        "cards=<ID[+N][xN],...> [relics=<ID[:m],...>] [relics_obtain=<ID,...>] [potions=<ID,...>] [potion_slots=<n>] [hp=<cur>[/max]]";

    public override string Description =>
        "Replaces run deck (with upgrades); inertly sets relics (with melted wax); sizes potion belt; adds potions; sets HP (dev-only, atomic).";

    public override bool IsNetworked => true;

    public override CmdResult Process(Player? issuingPlayer, string[] args)
    {
        if (!RunManager.Instance.IsInProgress)
        {
            return new CmdResult(success: false, "A run is currently not in progress!");
        }
        var player = issuingPlayer;
        if (player == null)
        {
            var runState = RunManager.Instance.DebugOnlyGetState();
            player = GameStateService.GetLocalPlayer(runState);
        }
        if (player == null)
        {
            return new CmdResult(success: false, "No player available for loadout.");
        }

        // ---- 解析 key=value 参数，并拒绝未知字段 ----
        string? cardsSpec = null;
        string? relicsSpec = null;
        string? relicsObtainSpec = null;
        string? potionsSpec = null;
        string? potionSlotsSpec = null;
        string? hpSpec = null;
        foreach (var arg in args)
        {
            var separator = arg.IndexOf('=');
            if (separator <= 0)
            {
                return new CmdResult(success: false, $"Malformed argument '{arg}' (expected key=value).");
            }
            var key = arg[..separator].ToLowerInvariant();
            var value = arg[(separator + 1)..];
            // 空值（如 "relics="、"potions="）表示未提供该字段，保持原有空操作
            // 语义；否则一个裸字段会悄悄变成“清空列表”。
            if (string.IsNullOrWhiteSpace(value))
            {
                continue;
            }
            switch (key)
            {
                case "cards": cardsSpec = value; break;
                case "relics": relicsSpec = value; break;
                case "relics_obtain": relicsObtainSpec = value; break;
                case "potions": potionsSpec = value; break;
                case "potion_slots": potionSlotsSpec = value; break;
                case "hp": hpSpec = value; break;
                default:
                    return new CmdResult(
                        success: false,
                        $"Unknown key '{key}' (expected cards/relics/relics_obtain/potions/potion_slots/hp).");
            }
        }
        if (cardsSpec == null && relicsSpec == null && relicsObtainSpec == null
            && potionsSpec == null && potionSlotsSpec == null && hpSpec == null)
        {
            return new CmdResult(
                success: false,
                "Nothing to do: pass at least one of cards=/relics=/relics_obtain=/potions=/potion_slots=/hp=.");
        }
        if (relicsSpec != null && relicsObtainSpec != null)
        {
            return new CmdResult(success: false, "relics= and relics_obtain= are mutually exclusive.");
        }

        // ---- 先校验全部参数，确保装载要么完整应用，要么完全不应用 ----
        IReadOnlyList<CardSpec>? cards = null;
        if (cardsSpec != null && !TryParseCards(cardsSpec, out cards, out var cardsError))
        {
            return new CmdResult(success: false, cardsError);
        }
        IReadOnlyList<RelicSpec>? relics = null;
        if (relicsSpec != null && !TryParseRelics(relicsSpec, out relics, out var relicsError))
        {
            return new CmdResult(success: false, relicsError);
        }
        IReadOnlyList<RelicModel>? relicsObtain = null;
        if (relicsObtainSpec != null && !TryParseRelicsObtain(relicsObtainSpec, out relicsObtain, out var obtainError))
        {
            return new CmdResult(success: false, obtainError);
        }
        IReadOnlyList<PotionModel>? potions = null;
        if (potionsSpec != null && !TryParsePotions(potionsSpec, out potions, out var potionsError))
        {
            return new CmdResult(success: false, potionsError);
        }
        int? potionSlots = null;
        if (potionSlotsSpec != null)
        {
            if (!int.TryParse(potionSlotsSpec, out var slots) || slots < 1 || slots > MaxPotionSlots)
            {
                return new CmdResult(
                    success: false, $"potion_slots must be an int in 1-{MaxPotionSlots}, got '{potionSlotsSpec}'.");
            }
            potionSlots = slots;
        }
        HpSpec? hp = null;
        if (hpSpec != null)
        {
            var match = HpToken.Match(hpSpec);
            if (!match.Success)
            {
                return new CmdResult(
                    success: false, $"Malformed hp spec '{hpSpec}' (expected <cur>[/<max>] with ints >= 1).");
            }
            var current = int.Parse(match.Groups["cur"].Value);
            int? max = match.Groups["max"].Success ? int.Parse(match.Groups["max"].Value) : null;
            // 未显式提供最大生命值时，当前生命值仍受玩家实时上限约束。
            if (current > (max ?? player.Creature.MaxHp))
            {
                return new CmdResult(
                    success: false, $"hp current {current} exceeds max {(max ?? player.Creature.MaxHp)}.");
            }
            hp = new HpSpec(current, max);
        }

        var spec = new LoadoutSpec(cards ?? Array.Empty<CardSpec>(), relics, relicsObtain,
            potions ?? Array.Empty<PotionModel>(), potionSlots, hp);
        var task = ApplyAsync(player, spec);
        return new CmdResult(task, success: true, Describe(spec));
    }

    private static bool TryParseCards(string spec, out IReadOnlyList<CardSpec>? cards, out string? error)
    {
        var parsed = new List<CardSpec>();
        var unknown = new List<string>();
        foreach (var token in Split(spec))
        {
            var match = CardToken.Match(token);
            if (!match.Success)
            {
                unknown.Add(token);
                continue;
            }
            var entry = match.Groups["id"].Value.ToUpperInvariant();
            var template = ModelDb.AllCards.FirstOrDefault(card => card.Id.Entry == entry);
            if (template == null)
            {
                unknown.Add(entry);
                continue;
            }
            var upgrade = match.Groups["up"].Success ? int.Parse(match.Groups["up"].Value) : 0;
            if (upgrade > template.MaxUpgradeLevel)
            {
                unknown.Add($"{entry}+{upgrade} (max upgrade level {template.MaxUpgradeLevel})");
                continue;
            }
            var count = match.Groups["count"].Success ? int.Parse(match.Groups["count"].Value) : 1;
            parsed.Add(new CardSpec(template, upgrade, count));
        }
        return Collect(parsed, unknown, "cards", out cards, out error);
    }

    private static bool TryParseRelics(string spec, out IReadOnlyList<RelicSpec>? relics, out string? error)
    {
        var parsed = new List<RelicSpec>();
        var unknown = new List<string>();
        foreach (var token in Split(spec))
        {
            var match = RelicToken.Match(token);
            if (!match.Success)
            {
                unknown.Add(token);
                continue;
            }
            var entry = match.Groups["id"].Value.ToUpperInvariant();
            var template = ModelDb.AllRelics.FirstOrDefault(relic => relic.Id.Entry == entry);
            if (template == null)
            {
                unknown.Add(entry);
                continue;
            }
            var melted = token.Contains(":m");
            parsed.Add(new RelicSpec(template, melted));
        }
        return Collect(parsed, unknown, "relics", out relics, out error);
    }

    private static bool TryParseRelicsObtain(string spec, out IReadOnlyList<RelicModel>? relics, out string? error)
    {
        var parsed = new List<RelicModel>();
        var unknown = new List<string>();
        foreach (var token in Split(spec))
        {
            var entry = token.ToUpperInvariant();
            var template = ModelDb.AllRelics.FirstOrDefault(relic => relic.Id.Entry == entry);
            if (template == null)
            {
                unknown.Add(entry);
            }
            else
            {
                parsed.Add(template);
            }
        }
        return Collect(parsed, unknown, "relics", out relics, out error);
    }

    private static bool TryParsePotions(string spec, out IReadOnlyList<PotionModel>? potions, out string? error)
    {
        var parsed = new List<PotionModel>();
        var unknown = new List<string>();
        foreach (var token in Split(spec))
        {
            var entry = token.ToUpperInvariant();
            var template = ModelDb.AllPotions.FirstOrDefault(potion => potion.Id.Entry == entry);
            if (template == null)
            {
                unknown.Add(entry);
            }
            else
            {
                parsed.Add(template);
            }
        }
        return Collect(parsed, unknown, "potions", out potions, out error);
    }

    private static bool Collect<T>(
        List<T> parsed, List<string> unknown, string what, out IReadOnlyList<T>? result, out string? error)
    {
        if (unknown.Count > 0)
        {
            result = null;
            error = $"Unknown or invalid {what}: " + string.Join(", ", unknown);
            return false;
        }
        result = parsed;
        error = null;
        return true;
    }

    private static IEnumerable<string> Split(string spec)
    {
        return spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    private static async Task ApplyAsync(Player player, LoadoutSpec spec)
    {
        // 开发辅助命令不能让返回任务进入故障态，否则 HTTP 层会统一映射为 500；
        // 这里记录错误，并交给状态核对判断执行结果。
        try
        {
            await ApplyDeck(player, spec.Cards);
            if (spec.Relics != null)
            {
                ApplyRelicsInert(player, spec.Relics);
            }
            else if (spec.RelicsObtain != null)
            {
                await ApplyRelicsObtained(player, spec.RelicsObtain);
            }
            if (spec.PotionSlots is int slots)
            {
                ApplyPotionSlots(player, slots);
            }
            await ApplyPotions(player, spec.Potions);
            if (spec.Hp != null)
            {
                await ApplyHp(player, spec.Hp);
            }
        }
        catch (Exception ex)
        {
            Log.Error($"[STS2AIAgent.loadout] apply failed: {ex}");
        }
    }

    private static async Task ApplyDeck(Player player, IReadOnlyList<CardSpec> cards)
    {
        if (cards.Count == 0)
        {
            return;
        }
        foreach (var existing in PileType.Deck.GetPile(player).Cards.ToList())
        {
            await CardPileCmd.RemoveFromDeck(existing, showPreview: false);
        }
        var runState = RunManager.Instance.DebugOnlyGetState()
            ?? throw new InvalidOperationException("No run state available for loadout.");
        foreach (var card in cards)
        {
            for (var i = 0; i < card.Count; i++)
            {
                var mutable = runState.CreateCard(card.Template, player);
                await CardPileCmd.Add(mutable, PileType.Deck, skipVisuals: true);
                for (var up = 0; up < card.UpgradeLevel; up++)
                {
                    // 使用 None 预览样式可避免无头场景创建升级特效节点；该 API
                    // 同时写入牌堆的 UpgradedCards 历史，与真实锻造行为一致。
                    CardCmd.Upgrade(mutable, CardPreviewStyle.None);
                }
            }
        }
    }

    private static void ApplyRelicsInert(Player player, IReadOnlyList<RelicSpec> relics)
    {
        // 有意采用存档读写式修改：静默调用 Add/RemoveRelicInternal，跳过拾取效果、
        // UI 动画和随机袋记账，避免 RelicCmd.Obtain 污染预设战斗。
        foreach (var existing in player.Relics.ToList())
        {
            player.RemoveRelicInternal(existing, silent: true);
        }
        foreach (var relic in relics)
        {
            var mutable = relic.Template.ToMutable();
            if (relic.Melted)
            {
                // ":m" 表示处于耗尽状态的玩具箱蜡制副本。蜡制状态属于实例标记，
                // 而不是遗物类型属性，因此必须先镀蜡，再执行融化。
                mutable.IsWax = true;
            }
            player.AddRelicInternal(mutable, silent: true);
            if (relic.Melted)
            {
                player.MeltRelicInternal(mutable);
            }
        }
    }

    private static async Task ApplyRelicsObtained(Player player, IReadOnlyList<RelicModel> relics)
    {
        foreach (var existing in player.Relics.ToList())
        {
            await RelicCmd.Remove(existing);
        }
        foreach (var template in relics)
        {
            await RelicCmd.Obtain(template.ToMutable(), player);
        }
    }

    private static void ApplyPotionSlots(Player player, int slots)
    {
        var delta = slots - player.PotionSlots.Count;
        if (delta > 0)
        {
            player.AddToMaxPotionCount(delta);
        }
        else if (delta < 0)
        {
            player.SubtractFromMaxPotionCount(-delta);
        }
    }

    private static async Task ApplyPotions(Player player, IReadOnlyList<PotionModel> potions)
    {
        for (var i = 0; i < potions.Count; i++)
        {
            if (!player.HasOpenPotionSlots)
            {
                Log.Warn($"[STS2AIAgent.loadout] potion belt full; skipped remaining {potions.Count - i} potions");
                break;
            }
            await PotionCmd.TryToProcure(potions[i].ToMutable(), player);
        }
    }

    private static async Task ApplyHp(Player player, HpSpec hp)
    {
        // 先调整生命上限，避免 CurrentHp 在中间状态短暂越界。
        if (hp.Max is int max)
        {
            await CreatureCmd.SetMaxHp(player.Creature, max);
        }
        await CreatureCmd.SetCurrentHp(player.Creature, hp.Current);
    }

    private static string Describe(LoadoutSpec spec)
    {
        var parts = new List<string>();
        if (spec.Cards.Count > 0)
        {
            var upgraded = spec.Cards.Sum(card => card.UpgradeLevel > 0 ? card.Count : 0);
            parts.Add($"deck={spec.Cards.Sum(card => card.Count)} cards" + (upgraded > 0 ? $" ({upgraded} upgraded)" : ""));
        }
        if (spec.Relics != null)
        {
            var melted = spec.Relics.Count(relic => relic.Melted);
            parts.Add($"relics={spec.Relics.Count}" + (melted > 0 ? $" ({melted} melted)" : ""));
        }
        if (spec.RelicsObtain != null)
        {
            parts.Add($"relics_obtain={spec.RelicsObtain.Count}");
        }
        if (spec.Potions.Count > 0)
        {
            parts.Add($"potions={spec.Potions.Count}");
        }
        if (spec.PotionSlots is int slots)
        {
            parts.Add($"potion_slots={slots}");
        }
        if (spec.Hp != null)
        {
            parts.Add($"hp={spec.Hp.Current}/{spec.Hp.Max?.ToString() ?? "keep"}");
        }
        return "Loadout applied: " + string.Join(", ", parts);
    }
}
