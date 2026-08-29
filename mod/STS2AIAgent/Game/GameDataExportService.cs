using System.Reflection;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Encounters;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Monsters;
using MegaCrit.Sts2.Core.Rooms;

namespace STS2AIAgent.Game;

/// <summary>
/// 从当前固定版本的 <see cref="ModelDb"/> 导出可审计的游戏知识集合。
/// </summary>
internal static class GameDataExportService
{
    private static readonly Regex CardMarkupRegex = new(@"\[(?:/?[^\]]+)\]", RegexOptions.Compiled);
    private static readonly Regex CardWhitespaceRegex = new(@"\s+", RegexOptions.Compiled);

    /// <summary>
    /// 按稳定集合名导出一类游戏知识。
    /// </summary>
    /// <param name="collection">HTTP 数据端点传入的集合名。</param>
    /// <returns>可由 JSON 序列化器直接编码的集合对象。</returns>
    /// <exception cref="KeyNotFoundException">集合名不受支持。</exception>
    public static object ExportCollection(string collection)
    {
        return collection.Trim().ToLowerInvariant() switch
        {
            "acts" => ExportActs(),
            "cards" => ExportCards(),
            "relics" => ExportRelics(),
            "monsters" => ExportMonsters(),
            "potions" => ExportPotions(),
            "enchantments" => ExportEnchantments(),
            "events" => ExportEvents(),
            "encounters" => ExportEncounters(),
            "powers" => ExportPowers(),
            "characters" => ExportCharacters(),
            "keywords" => ExportKeywords(),
            _ => throw new KeyNotFoundException($"Unknown data collection: {collection}")
        };
    }

    /// <summary>
    /// 导出每张地图的前期弱遭遇、常规遭遇、精英和 Boss 池。
    /// </summary>
    /// <returns>按幕序与稳定 ID 排序的地图知识数组。</returns>
    private static object ExportActs()
    {
        return ModelDb.Acts
            .OrderBy(act => act.Index)
            .ThenBy(act => act.Id.Entry, StringComparer.Ordinal)
            .Select(act => new
            {
                id = act.Id.Entry,
                name = act.Title.GetFormattedText(),
                index = act.Index + 1,
                is_default = act.IsDefault,
                weak_encounters = BuildEncounterReferences(act.AllWeakEncounters),
                regular_encounters = BuildEncounterReferences(act.AllRegularEncounters),
                elite_encounters = BuildEncounterReferences(act.AllEliteEncounters),
                boss_encounters = BuildEncounterReferences(act.AllBossEncounters)
            })
            .ToArray();
    }

    /// <summary>
    /// 把遭遇模型序列转换为稳定 ID 与本地化名称引用。
    /// </summary>
    /// <param name="encounters">同一地图、同一房间等级的遭遇模型。</param>
    /// <returns>按稳定 ID 排序且去重的遭遇引用数组。</returns>
    private static object[] BuildEncounterReferences(IEnumerable<EncounterModel> encounters)
    {
        return encounters
            .GroupBy(encounter => encounter.Id.Entry, StringComparer.Ordinal)
            .Select(group => group.First())
            .OrderBy(encounter => encounter.Id.Entry, StringComparer.Ordinal)
            .Select(encounter => (object)new
            {
                id = encounter.Id.Entry,
                name = encounter.Title.GetFormattedText()
            })
            .ToArray();
    }

    /// <summary>
    /// 导出除 <see cref="CardKeyword.None"/> 外的卡牌关键词及其本地化释义。
    /// </summary>
    /// <returns>关键词知识数组。</returns>
    private static object ExportKeywords()
    {
        return Enum.GetValues<CardKeyword>()
            .Where(keyword => keyword != CardKeyword.None)
            .Select(keyword =>
            {
                var key = keyword.ToString().ToUpperInvariant();
                return new
                {
                    id = key,
                    name = LocString.GetIfExists(
                        "card_keywords", key + ".title")?.GetFormattedText(),
                    description = LocString.GetIfExists(
                        "card_keywords", key + ".description")?.GetFormattedText()
                };
            })
            .ToArray();
    }

    /// <summary>
    /// 导出卡牌基础属性、已解析文本、动态变量和真实升级预览。
    /// </summary>
    /// <returns>卡牌知识数组。</returns>
    private static object ExportCards()
    {
        return ModelDb.AllCards
            .OrderBy(card => card.Id.Entry, StringComparer.Ordinal)
            .Select(card =>
            {
                var dynamicValues = BuildCardDynamicValuePayloads(card);
                return new
                {
                    id = card.Id.Entry,
                    name = card.Title,
                    description = GetResolvedCardRulesText(card),
                    description_raw = GetCardRulesText(card),
                    type = card.Type.ToString(),
                    rarity = card.Rarity.ToString(),
                    target = card.TargetType.ToString(),
                    cost = card.EnergyCost.Canonical,
                    is_x_cost = card.EnergyCost.CostsX,
                    star_cost = card.CanonicalStarCost >= 0 ? (int?)card.CanonicalStarCost : null,
                    is_x_star_cost = card.HasStarCostX,
                    color = GetCardColor(card),
                    damage = FindDynamicValue(dynamicValues, "damage"),
                    block = FindDynamicValue(dynamicValues, "block"),
                    keywords = card.Keywords.Select(keyword => keyword.ToString()).OrderBy(value => value, StringComparer.Ordinal).ToArray(),
                    tags = card.Tags.Select(tag => tag.ToString()).OrderBy(value => value, StringComparer.Ordinal).ToArray(),
                    max_upgrade_level = card.MaxUpgradeLevel,
                    vars = dynamicValues.Select(value => new
                    {
                        name = value.Name,
                        base_value = value.BaseValue,
                        current_value = value.CurrentValue,
                        enchanted_value = value.EnchantedValue,
                        is_modified = value.IsModified,
                        was_just_upgraded = value.WasJustUpgraded
                    }).ToArray(),
                    upgrade = BuildCardUpgradePreview(card)
                };
            })
            .ToArray();
    }

    /// <summary>
    /// 导出遗物的本地化效果、稀有度、池和融化状态。
    /// </summary>
    /// <returns>遗物知识数组。</returns>
    private static object ExportRelics()
    {
        return ModelDb.AllRelics
            .OrderBy(relic => relic.Id.Entry, StringComparer.Ordinal)
            .Select(relic => new
            {
                id = relic.Id.Entry,
                name = relic.Title.GetFormattedText(),
                description = GetDynamicFormattedTextProperty(relic, "DynamicDescription", "Description"),
                rarity = relic.Rarity.ToString(),
                pool = relic.Pool.ToString().ToLowerInvariant(),
                is_melted = relic.IsMelted
            })
            .ToArray();
    }

    /// <summary>
    /// 导出药水的本地化效果、稀有度、使用时机和目标类型。
    /// </summary>
    /// <returns>药水知识数组。</returns>
    private static object ExportPotions()
    {
        return ModelDb.AllPotions
            .OrderBy(potion => potion.Id.Entry, StringComparer.Ordinal)
            .Select(potion => new
            {
                id = potion.Id.Entry,
                name = potion.Title.GetFormattedText(),
                description = GetDynamicFormattedTextProperty(potion, "DynamicDescription", "Description"),
                rarity = potion.Rarity.ToString(),
                pool = potion.Pool.ToString().ToLowerInvariant(),
                usage = potion.Usage.ToString(),
                target_type = potion.TargetType.ToString()
            })
            .ToArray();
    }

    /// <summary>
    /// 导出普通事件、远古者和结局事件的初始页面事实。
    /// </summary>
    /// <returns>事件知识数组。</returns>
    private static object ExportEvents()
    {
        return ModelDb.AllEvents
            .Concat<EventModel>(ModelDb.AllAncients)
            .Append(ModelDb.Event<TheArchitect>())
            .DistinctBy(eventModel => eventModel.Id.Entry)
            .OrderBy(eventModel => eventModel.Id.Entry, StringComparer.Ordinal)
            .Select(eventModel => new
            {
                id = eventModel.Id.Entry,
                name = eventModel.Title.GetFormattedText(),
                type = eventModel is AncientEventModel ? "Ancient" : "Event",
                act = ResolveEventAct(eventModel),
                description_kind = eventModel is AncientEventModel
                    ? "dynamic_dialogue"
                    : "initial_page",
                description = eventModel is AncientEventModel
                    ? string.Empty
                    : eventModel.InitialDescription.GetRawText(),
                options = BuildEventOptions(eventModel)
            })
            .ToArray();
    }

    /// <summary>
    /// 以一层样例数值导出附魔效果和附加卡面文本。
    /// </summary>
    /// <returns>附魔知识数组。</returns>
    private static object ExportEnchantments()
    {
        return ModelDb.DebugEnchantments
            .OrderBy(enchantment => enchantment.Id.Entry, StringComparer.Ordinal)
            .Select(enchantment =>
            {
                var mutable = enchantment.ToMutable();
                mutable.Amount = 1;
                mutable.RecalculateValues();
                var description = mutable.DynamicDescription.GetFormattedText();
                string? extraCardText = null;
                if (mutable.HasExtraCardText)
                {
                    var extra = new LocString(
                        "enchantments",
                        mutable.Id.Entry + ".extraCardText");
                    extra.Add("Amount", mutable.Amount);
                    extra.Add("TargetType", "None");
                    extra.Add("energyPrefix", EnergyIconHelper.GetPrefix(mutable));
                    mutable.DynamicVars.AddTo(extra);
                    extraCardText = extra.GetFormattedText();
                }
                return new
                {
                    id = enchantment.Id.Entry,
                    name = enchantment.Title.GetFormattedText(),
                    model_type = enchantment.GetType().FullName,
                    description,
                    extra_card_text = extraCardText,
                    is_stackable = enchantment.IsStackable,
                    show_amount = enchantment.ShowAmount,
                    sample_amount = 1
                };
            })
            .ToArray();
    }

    /// <summary>
    /// 以强度一为样例导出能力的已解析描述和叠加语义。
    /// </summary>
    /// <returns>能力知识数组。</returns>
    private static object ExportPowers()
    {
        return ModelDb.AllPowers
            .OrderBy(power => power.Id.Entry, StringComparer.Ordinal)
            .Select(power => new
            {
                id = power.Id.Entry,
                name = power.Title.GetFormattedText(),
                model_type = power.GetType().FullName,
                description = power.GetDumbHoverTip(1).Description,
                description_raw = power.Description.GetRawText(),
                sample_amount = 1,
                uses_amount = power.Description.GetRawText()
                    .Contains("{Amount}", StringComparison.Ordinal),
                type = power.Type.ToString(),
                stack_type = power.StackType.ToString(),
                allow_negative = power.AllowNegative
            })
            .ToArray();
    }

    /// <summary>
    /// 导出角色的初始属性、牌组、遗物和药水。
    /// </summary>
    /// <returns>角色知识数组。</returns>
    private static object ExportCharacters()
    {
        return ModelDb.AllCharacters
            .OrderBy(character => character.Id.Entry, StringComparer.Ordinal)
            .Select(character => new
            {
                id = character.Id.Entry,
                name = character.Title.GetFormattedText(),
                description = LocString.GetIfExists("characters", character.Id.Entry + ".description")?.GetFormattedText(),
                starting_hp = character.StartingHp,
                starting_gold = character.StartingGold,
                max_energy = character.MaxEnergy,
                orb_slots = character.BaseOrbSlotCount,
                gender = character.Gender.ToString(),
                color = character.CardPool.Title.ToLowerInvariant(),
                starting_deck = character.StartingDeck.Select(card => card.Id.Entry).ToArray(),
                starting_relics = character.StartingRelics.Select(relic => relic.Id.Entry).ToArray(),
                starting_potions = character.StartingPotions.Select(potion => potion.Id.Entry).ToArray()
            })
            .ToArray();
    }

    /// <summary>
    /// 导出可达怪物的生命范围、类型、招式和关联遭遇。
    /// </summary>
    /// <returns>怪物知识数组。</returns>
    private static object ExportMonsters()
    {
        return GetKnowledgeMonsters()
            .GroupBy(monster => monster.Id.Entry, StringComparer.Ordinal)
            .Select(group => group.First())
            .OrderBy(monster => monster.Id.Entry, StringComparer.Ordinal)
            .Select(monster => new
            {
                id = monster.Id.Entry,
                name = monster.Title.GetFormattedText(),
                model_type = monster.GetType().FullName,
                show_in_compendium = monster.ShouldShowInCompendium,
                type = ResolveMonsterType(monster),
                min_hp = monster.MinInitialHp,
                max_hp = monster.MaxInitialHp,
                moves = BuildMonsterMoves(monster),
                acts = ModelDb.Acts
                    .Where(act => act.AllMonsters.Any(candidate => candidate.Id == monster.Id))
                    .OrderBy(act => act.Index)
                    .Select(act => new
                    {
                        id = act.Id.Entry,
                        index = act.Index + 1,
                        name = act.Title.GetFormattedText()
                    })
                    .ToArray(),
                encounters = GetKnowledgeEncounters()
                    .Where(encounter => encounter.AllPossibleMonsters.Any(candidate => candidate.Id == monster.Id))
                    .Select(encounter => encounter.Id.Entry)
                    .OrderBy(value => value, StringComparer.Ordinal)
                    .ToArray(),
                damage_values = (object?)null,
                block_values = (object?)null
            })
            .ToArray();
    }

    /// <summary>
    /// 汇总地图遭遇怪物、数据库怪物和真实机制召唤的辅助生物。
    /// </summary>
    /// <returns>按稳定 ID 去重后的知识怪物序列。</returns>
    private static IEnumerable<MonsterModel> GetKnowledgeMonsters()
    {
        // 部分真实战斗生物由事件、遗物或角色机制召唤，因此不会出现在
        // EncounterModel.AllPossibleMonsters 中，必须显式补入。
        var auxiliaryCreatures = new MonsterModel?[]
        {
            GetRegisteredMonster<Byrdpip>(),
            GetRegisteredMonster<Osty>(),
            GetRegisteredMonster<PaelsLegion>(),
            GetRegisteredMonster<TheAdversaryMkOne>(),
            GetRegisteredMonster<TheAdversaryMkTwo>(),
            GetRegisteredMonster<TheAdversaryMkThree>()
        }.OfType<MonsterModel>();
        return GetKnowledgeEncounters()
            .SelectMany(encounter => encounter.AllPossibleMonsters)
            .Concat(ModelDb.Monsters)
            .Concat(auxiliaryCreatures)
            .GroupBy(monster => monster.Id.Entry, StringComparer.Ordinal)
            .Select(group => group.First());
    }

    /// <summary>
    /// 导出地图与事件战遭遇的属性和全部可能敌人类型。
    /// </summary>
    /// <returns>遭遇知识数组。</returns>
    private static object ExportEncounters()
    {
        return GetKnowledgeEncounters()
            .OrderBy(encounter => encounter.Id.Entry, StringComparer.Ordinal)
            .Select(encounter => new
            {
                id = encounter.Id.Entry,
                name = encounter.Title.GetFormattedText(),
                model_type = encounter.GetType().FullName,
                room_type = encounter.RoomType.ToString(),
                is_weak = encounter.IsWeak,
                is_debug = IsDebugEncounter(encounter),
                should_give_rewards = encounter.ShouldGiveRewards,
                monster_list_kind = "all_possible_types",
                tags = encounter.Tags
                    .Select(tag => tag.ToString())
                    .OrderBy(value => value, StringComparer.Ordinal)
                    .ToArray(),
                monsters = encounter.AllPossibleMonsters
                    .GroupBy(monster => monster.Id.Entry, StringComparer.Ordinal)
                    .Select(group => group.First())
                    .OrderBy(monster => monster.Id.Entry, StringComparer.Ordinal)
                    .Select(monster => new
                    {
                        id = monster.Id.Entry,
                        name = monster.Title.GetFormattedText()
                    })
                    .ToArray()
            })
            .ToArray();
    }

    /// <summary>
    /// 汇总地图遭遇和当前版本可真实到达的事件战遭遇。
    /// </summary>
    /// <returns>按稳定 ID 去重后的知识遭遇序列。</returns>
    private static IEnumerable<EncounterModel> GetKnowledgeEncounters()
    {
        // ModelDb.AllEncounters 只包含 Act 地图遭遇；事件战同样真实可达，因此按
        // 固定版本显式列举，避免遗漏它们或把所有内部测试遭遇一并放入。
        var legacyEventEncounters = new EncounterModel?[]
        {
            GetRegisteredEncounter<BattlewornDummyEventEncounter>(),
            GetRegisteredEncounter<DenseVegetationEventEncounter>(),
            GetRegisteredEncounter<FakeMerchantEventEncounter>(),
            GetRegisteredEncounter<MysteriousKnightEventEncounter>(),
            GetRegisteredEncounter<PunchOffEventEncounter>(),
            GetRegisteredEncounter<TheArchitectEventEncounter>()
        }.OfType<EncounterModel>();
        return ModelDb.AllEncounters
            .Concat(legacyEventEncounters)
            .GroupBy(encounter => encounter.Id.Entry, StringComparer.Ordinal)
            .Select(group => group.First());
    }

    /// <summary>
    /// 获取当前游戏版本实际注册的辅助怪物。
    /// </summary>
    /// <typeparam name="TMonster">跨版本源码中存在的怪物模型类型。</typeparam>
    /// <returns>已注册的规范怪物；当前版本未注册时返回空值。</returns>
    private static MonsterModel? GetRegisteredMonster<TMonster>()
        where TMonster : MonsterModel
    {
        return ModelDb.Contains(typeof(TMonster)) ? ModelDb.Monster<TMonster>() : null;
    }

    /// <summary>
    /// 获取当前游戏版本实际注册的旧版事件战遭遇。
    /// </summary>
    /// <typeparam name="TEncounter">跨版本源码中存在的遭遇模型类型。</typeparam>
    /// <returns>已注册的规范遭遇；当前版本未注册时返回空值。</returns>
    private static EncounterModel? GetRegisteredEncounter<TEncounter>()
        where TEncounter : EncounterModel
    {
        return ModelDb.Contains(typeof(TEncounter)) ? ModelDb.Encounter<TEncounter>() : null;
    }

    /// <summary>
    /// 将卡池映射为稳定的卡牌颜色标识。
    /// </summary>
    /// <param name="card">待判断的卡牌模型。</param>
    /// <returns>无色标识或小写卡池名称。</returns>
    private static string GetCardColor(CardModel card)
    {
        if (card.Pool.IsColorless)
        {
            return "colorless";
        }

        return card.Pool.Title.ToLowerInvariant();
    }

    /// <summary>
    /// 查找事件所属 Act；跨 Act 事件使用共享标识。
    /// </summary>
    /// <param name="eventModel">待定位的事件模型。</param>
    /// <returns>本地化 Act 名称或 <c>Shared</c>。</returns>
    private static string ResolveEventAct(EventModel eventModel)
    {
        var act = ModelDb.Acts.FirstOrDefault(candidate =>
            candidate.AllEvents.Contains(eventModel)
            || eventModel is AncientEventModel ancient
            && candidate.AllAncients.Contains(ancient));
        return act?.Title.GetFormattedText() ?? "Shared";
    }

    /// <summary>
    /// 以关联遭遇的最高房间等级推导怪物类型。
    /// </summary>
    /// <param name="monster">待分类的怪物模型。</param>
    /// <returns><c>Boss</c>、<c>Elite</c>、<c>Normal</c> 或 <c>Unknown</c>。</returns>
    private static string ResolveMonsterType(MonsterModel monster)
    {
        var roomType = GetKnowledgeEncounters()
            .Where(encounter => encounter.AllPossibleMonsters.Contains(monster))
            .Select(encounter => encounter.RoomType)
            .OrderByDescending(value => value)
            .FirstOrDefault();

        return roomType switch
        {
            RoomType.Boss => "Boss",
            RoomType.Elite => "Elite",
            RoomType.Monster => "Normal",
            _ => "Unknown"
        };
    }

    /// <summary>
    /// 通过怪物图鉴接口读取当前版本公开的招式名称。
    /// </summary>
    /// <param name="monster">待实例化的怪物模型。</param>
    /// <returns>去重后的招式标识与名称；游戏拒绝构造时返回空数组。</returns>
    private static object[] BuildMonsterMoves(MonsterModel monster)
    {
        try
        {
            var mutable = monster.ToMutable();
            mutable.SetUpForCombat();
            return mutable.GenerateBestiaryMoveList(null)
                .Select(move => new
                {
                    id = move.stateId ?? move.animId ?? move.displayName,
                    name = NormalizeCardRulesText(move.displayName)
                })
                .Where(move => !string.IsNullOrWhiteSpace(move.id) || !string.IsNullOrWhiteSpace(move.name))
                .DistinctBy(move => (move.id, move.name))
                .ToArray<object>();
        }
        catch
        {
            return Array.Empty<object>();
        }
    }

    /// <summary>
    /// 从事件游戏信息键中提取初始页面选项模板。
    /// </summary>
    /// <param name="eventModel">待读取的事件模型。</param>
    /// <returns>选项 ID、标题和描述模板；读取失败时返回空数组。</returns>
    private static object[] BuildEventOptions(EventModel eventModel)
    {
        try
        {
            var prefix = $"{eventModel.Id.Entry}.pages.INITIAL.options.";
            return eventModel.GameInfoOptions
                .Select(locString => locString.LocEntryKey)
                .Select(key => TrimKnownSuffix(key, ".title"))
                .Select(key => TrimKnownSuffix(key, ".description"))
                .Where(key => key.StartsWith(prefix, StringComparison.Ordinal))
                .Distinct(StringComparer.Ordinal)
                .Select(key => new
                {
                    id = ExtractKeySegment(key, prefix),
                    title = eventModel.GetOptionTitle(key)?.GetRawText() ?? string.Empty,
                    description = eventModel.GetOptionDescription(key)?.GetRawText() ?? string.Empty
                })
                .ToArray<object>();
        }
        catch
        {
            return Array.Empty<object>();
        }
    }

    /// <summary>
    /// 从本地化键中截取初始事件选项的稳定 ID。
    /// </summary>
    /// <param name="key">完整本地化键。</param>
    /// <param name="prefix">事件初始选项键前缀。</param>
    /// <returns>首个后缀段；前缀不匹配时原样返回。</returns>
    private static string ExtractKeySegment(string key, string prefix)
    {
        if (!key.StartsWith(prefix, StringComparison.Ordinal))
        {
            return key;
        }

        var suffix = key[prefix.Length..];
        var separator = suffix.IndexOf('.');
        return separator >= 0 ? suffix[..separator] : suffix;
    }

    /// <summary>
    /// 仅在精确匹配时移除已知本地化键后缀。
    /// </summary>
    /// <param name="value">待处理的本地化键。</param>
    /// <param name="suffix">允许移除的后缀。</param>
    /// <returns>移除后缀后的键或原值。</returns>
    private static string TrimKnownSuffix(string value, string suffix)
    {
        return value.EndsWith(suffix, StringComparison.Ordinal)
            ? value[..^suffix.Length]
            : value;
    }

    /// <summary>
    /// 在独立可变副本上执行一次真实升级并导出升级后事实。
    /// </summary>
    /// <param name="card">待升级预览的基础卡牌模型。</param>
    /// <returns>升级后描述、费用和动态变量；不可升级或失败时返回空值。</returns>
    private static object? BuildCardUpgradePreview(CardModel card)
    {
        if (!card.IsUpgradable)
        {
            return null;
        }

        try
        {
            // 升级描述预览不包含费用等非文案变化，而且旧实现会临时修改
            // ModelDb 共享单例。用独立 mutable 副本执行真实升级，完整导出结果。
            var upgraded = card.ToMutable();
            upgraded.UpgradeInternal();
            var dynamicValues = BuildCardDynamicValuePayloads(upgraded);
            return new
            {
                level = upgraded.CurrentUpgradeLevel,
                description = GetResolvedCardRulesText(upgraded),
                cost = upgraded.EnergyCost.CostsX
                    ? 0
                    : upgraded.EnergyCost.GetWithModifiers(CostModifiers.Local),
                is_x_cost = upgraded.EnergyCost.CostsX,
                star_cost = upgraded.BaseStarCost >= 0
                    ? (int?)upgraded.BaseStarCost
                    : null,
                is_x_star_cost = upgraded.HasStarCostX,
                damage = FindDynamicValue(dynamicValues, "damage"),
                block = FindDynamicValue(dynamicValues, "block"),
                vars = dynamicValues.Select(value => new
                {
                    name = value.Name,
                    base_value = value.BaseValue,
                    current_value = value.CurrentValue,
                    enchanted_value = value.EnchantedValue,
                    is_modified = value.IsModified,
                    was_just_upgraded = value.WasJustUpgraded
                }).ToArray()
            };
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// 按不区分大小写的动态变量名称读取当前数值。
    /// </summary>
    /// <param name="values">卡牌动态变量快照。</param>
    /// <param name="name">目标变量名。</param>
    /// <returns>命中的当前值；不存在时返回空值。</returns>
    private static int? FindDynamicValue(CardDynamicValueInfo[] values, string name)
    {
        foreach (var value in values)
        {
            if (value.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            {
                return value.CurrentValue;
            }
        }

        return null;
    }

    /// <summary>
    /// 计算普通预览上下文中的卡牌动态变量快照。
    /// </summary>
    /// <param name="card">基础或升级后的卡牌模型。</param>
    /// <returns>按变量名排序的动态变量数组；无法计算时返回空数组。</returns>
    private static CardDynamicValueInfo[] BuildCardDynamicValuePayloads(CardModel? card)
    {
        if (card == null)
        {
            return Array.Empty<CardDynamicValueInfo>();
        }

        try
        {
            var previewSet = card.DynamicVars.Clone(card);
            card.UpdateDynamicVarPreview(CardPreviewMode.Normal, card.CurrentTarget, previewSet);

            return previewSet.Values
                .Select(dynamicVar => new CardDynamicValueInfo(
                    dynamicVar.Name,
                    (int)dynamicVar.BaseValue,
                    (int)dynamicVar.PreviewValue,
                    (int)dynamicVar.EnchantedValue,
                    (int)dynamicVar.PreviewValue != (int)dynamicVar.BaseValue
                        || (int)dynamicVar.EnchantedValue != (int)dynamicVar.BaseValue,
                    dynamicVar.WasJustUpgraded))
                .OrderBy(value => value.Name, StringComparer.Ordinal)
                .ToArray();
        }
        catch
        {
            return Array.Empty<CardDynamicValueInfo>();
        }
    }

    /// <summary>
    /// 从不同卡牌模型版本可能使用的成员中读取原始规则文本。
    /// </summary>
    /// <param name="card">待读取的卡牌模型。</param>
    /// <returns>去除显示标记并压缩空白后的原始规则文本。</returns>
    private static string GetCardRulesText(CardModel? card)
    {
        if (card == null)
        {
            return string.Empty;
        }

        foreach (var memberName in new[]
        {
            "Description",
            "RulesText",
            "Body",
            "Text",
            "RawText",
            "DescriptionText"
        })
        {
            var text = TryReadCardTextMember(card, memberName);
            if (!string.IsNullOrWhiteSpace(text))
            {
                return NormalizeCardRulesText(text);
            }
        }

        return string.Empty;
    }

    /// <summary>
    /// 在当前动态变量与牌堆上下文中解析卡牌规则文本。
    /// </summary>
    /// <param name="card">待解析的卡牌模型。</param>
    /// <returns>解析后的规则文本；失败时回退到原始规则文本。</returns>
    private static string GetResolvedCardRulesText(CardModel? card)
    {
        if (card == null)
        {
            return string.Empty;
        }

        try
        {
            card.UpdateDynamicVarPreview(CardPreviewMode.Normal, card.CurrentTarget, card.DynamicVars);
            var pileType = card.Pile?.Type ?? PileType.None;
            var resolved = card.GetDescriptionForPile(pileType, card.CurrentTarget);
            if (!string.IsNullOrWhiteSpace(resolved))
            {
                return NormalizeCardRulesText(resolved);
            }
        }
        catch
        {
        }

        return GetCardRulesText(card);
    }

    /// <summary>
    /// 兼容不同游戏构建，通过反射尝试读取一个卡牌文本成员。
    /// </summary>
    /// <param name="instance">卡牌模型实例。</param>
    /// <param name="memberName">候选属性或字段名。</param>
    /// <returns>可读取的原始文本；成员不存在或读取失败时返回空串。</returns>
    private static string TryReadCardTextMember(object instance, string memberName)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        try
        {
            var property = instance.GetType().GetProperty(memberName, flags);
            if (property != null)
            {
                return TryCoerceRawText(property.GetValue(instance));
            }

            var field = instance.GetType().GetField(memberName, flags);
            if (field != null)
            {
                return TryCoerceRawText(field.GetValue(instance));
            }
        }
        catch
        {
        }

        return string.Empty;
    }

    /// <summary>
    /// 将字符串或本地化文本对象尽力转换为格式化文本。
    /// </summary>
    /// <param name="value">字符串、本地化对象或其他可显示值。</param>
    /// <returns>格式化文本；空值或转换失败时返回空串。</returns>
    private static string TryCoerceText(object? value)
    {
        if (value == null)
        {
            return string.Empty;
        }

        if (value is string text)
        {
            return text;
        }

        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        var valueType = value.GetType();

        try
        {
            var getFormattedText = valueType.GetMethod("GetFormattedText", flags, null, Type.EmptyTypes, null);
            if (getFormattedText != null && getFormattedText.ReturnType == typeof(string))
            {
                return getFormattedText.Invoke(value, null) as string ?? string.Empty;
            }
        }
        catch
        {
        }

        try
        {
            var getRawText = valueType.GetMethod("GetRawText", flags, null, Type.EmptyTypes, null);
            if (getRawText != null && getRawText.ReturnType == typeof(string))
            {
                return getRawText.Invoke(value, null) as string ?? string.Empty;
            }
        }
        catch
        {
        }

        return value.ToString() ?? string.Empty;
    }

    /// <summary>
    /// 优先保留本地化模板的原始文本，再回退到格式化文本。
    /// </summary>
    /// <param name="value">字符串、本地化对象或其他可显示值。</param>
    /// <returns>原始文本或兼容回退文本。</returns>
    private static string TryCoerceRawText(object? value)
    {
        if (value == null)
        {
            return string.Empty;
        }

        if (value is string text)
        {
            return text;
        }

        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        var valueType = value.GetType();
        try
        {
            var getRawText = valueType.GetMethod("GetRawText", flags, null, Type.EmptyTypes, null);
            if (getRawText != null && getRawText.ReturnType == typeof(string))
            {
                return getRawText.Invoke(value, null) as string ?? string.Empty;
            }
        }
        catch
        {
        }

        return TryCoerceText(value);
    }

    /// <summary>
    /// 移除卡牌显示标记并把连续空白压缩为单个空格。
    /// </summary>
    /// <param name="value">游戏返回的卡牌规则文本。</param>
    /// <returns>适合知识导出的纯文本。</returns>
    private static string NormalizeCardRulesText(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        var normalized = CardMarkupRegex.Replace(value, string.Empty);
        normalized = CardWhitespaceRegex.Replace(normalized, " ");
        return normalized.Trim();
    }

    /// <summary>
    /// 安全读取游戏模型的公开属性。
    /// </summary>
    /// <param name="target">目标游戏模型。</param>
    /// <param name="propertyName">属性名。</param>
    /// <returns>属性值；属性不存在或访问失败时返回空值。</returns>
    private static object? GetReflectedProperty(object target, string propertyName)
    {
        try
        {
            return target.GetType().GetProperty(propertyName)?.GetValue(target);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// 判断遭遇是否只用于调试、测试或已废弃的历史记录。
    /// </summary>
    /// <param name="encounter">待导出的遭遇模型。</param>
    /// <returns>旧版调试标记或新版 Mock 标记成立，以及遭遇已废弃时返回真。</returns>
    private static bool IsDebugEncounter(EncounterModel encounter)
    {
        var legacyValue = GetReflectedProperty(encounter, "IsDebugEncounter");
        if (legacyValue is bool legacyDebug)
        {
            return legacyDebug;
        }

        return GetReflectedProperty(encounter, "IsMock") is true
            || encounter is DeprecatedEncounter;
    }

    /// <summary>
    /// 读取游戏模型属性并转换为格式化文本。
    /// </summary>
    /// <param name="target">目标游戏模型。</param>
    /// <param name="propertyName">文本属性名。</param>
    /// <returns>格式化文本；属性不可用时返回空值。</returns>
    private static string? GetReflectedFormattedTextProperty(object target, string propertyName)
    {
        var value = GetReflectedProperty(target, propertyName);
        return value == null ? null : TryCoerceText(value);
    }

    /// <summary>
    /// 按优先顺序读取第一个非空的动态描述属性。
    /// </summary>
    /// <param name="target">目标游戏模型。</param>
    /// <param name="propertyNames">按兼容优先级排列的候选属性名。</param>
    /// <returns>第一个非空格式化文本；全部不可用时返回空值。</returns>
    private static string? GetDynamicFormattedTextProperty(object target, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            var value = GetReflectedFormattedTextProperty(target, propertyName);
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value;
            }
        }

        return null;
    }

    /// <summary>
    /// 保存一项卡牌动态变量在普通预览中的完整数值状态。
    /// </summary>
    /// <param name="Name">动态变量名称。</param>
    /// <param name="BaseValue">基础数值。</param>
    /// <param name="CurrentValue">当前预览数值。</param>
    /// <param name="EnchantedValue">附魔后的预览数值。</param>
    /// <param name="IsModified">当前值或附魔值是否偏离基础值。</param>
    /// <param name="WasJustUpgraded">该变量是否刚因升级发生变化。</param>
    private readonly record struct CardDynamicValueInfo(
        string Name,
        int BaseValue,
        int CurrentValue,
        int EnchantedValue,
        bool IsModified,
        bool WasJustUpgraded);
}
