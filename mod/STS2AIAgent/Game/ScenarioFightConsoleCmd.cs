using System.Reflection;
using MegaCrit.Sts2.Core.DevConsole;
using MegaCrit.Sts2.Core.DevConsole.ConsoleCommands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Exceptions;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2AIAgent.Game;

/// <summary>
/// 以游戏原生遭遇 RNG 公式进入指定战斗：
///   scenariofight CULTISTS_NORMAL floor=7
/// <c>floor</c> 表示原战斗的总层数，只用于计算遭遇 RNG；命令不会伪造地图
/// 历史。与游戏自带 <c>fight</c> 命令不同，此命令不会按当前时间随机化遭遇，
/// 因而相同局种子、总层数和遭遇 ID 会生成相同的敌人组成与生命值。
/// </summary>
public sealed class ScenarioFightConsoleCmd : AbstractConsoleCmd
{
    private static readonly FieldInfo? EncounterRngField = typeof(EncounterModel).GetField(
        "_rng",
        BindingFlags.Instance | BindingFlags.NonPublic);

    public override string CmdName => "scenariofight";

    public override string Args => "<id:string> [floor=<total_floor:int>]";

    public override string Description =>
        "Jumps to an encounter with deterministic seed/floor encounter RNG (dev-only).";

    public override bool IsNetworked => true;

    /// <summary>
    /// 校验遭遇和总层数，注入确定性遭遇 RNG 后进入调试战斗房间。
    /// </summary>
    /// <param name="issuingPlayer">发出命令的玩家；单机 HTTP 调用时可为空。</param>
    /// <param name="args">遭遇 ID 与可选的原总层数。</param>
    /// <returns>包含房间切换任务的控制台执行结果。</returns>
    public override CmdResult Process(Player? issuingPlayer, string[] args)
    {
        if (args.Length is < 1 or > 2)
        {
            return new CmdResult(success: false, $"Usage: {CmdName} {Args}");
        }
        if (!RunManager.Instance.IsInProgress)
        {
            return new CmdResult(success: false, "A run is currently not in progress!");
        }

        var runState = RunManager.Instance.DebugOnlyGetState();
        if (runState == null)
        {
            return new CmdResult(success: false, "Run state is unavailable.");
        }

        var floor = runState.TotalFloor;
        if (args.Length == 2)
        {
            const string prefix = "floor=";
            if (!args[1].StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
                || !int.TryParse(args[1][prefix.Length..], out floor)
                || floor < 0)
            {
                return new CmdResult(success: false, "floor must be an int greater than or equal to 0.");
            }
        }

        var modelId = new ModelId(
            ModelId.SlugifyCategory<EncounterModel>(),
            args[0].ToUpperInvariant());
        EncounterModel encounter;
        try
        {
            encounter = ModelDb.GetById<EncounterModel>(modelId).ToMutable();
        }
        catch (ModelNotFoundException)
        {
            return new CmdResult(success: false, $"Encounter '{modelId.Entry}' not found");
        }

        if (EncounterRngField == null)
        {
            return new CmdResult(success: false, "Encounter RNG field is unavailable.");
        }

        // 与 EncounterModel.GenerateMonstersWithSlots 使用完全相同的公式，只把
        // 当前调试局层数替换为场景记录的原总层数。
        var encounterSeed = unchecked((uint)(
            (int)runState.Rng.Seed
            + floor
            + StringHelper.GetDeterministicHashCode(encounter.Id.Entry)));
        EncounterRngField.SetValue(encounter, new Rng(encounterSeed));

        var task = RunManager.Instance.EnterRoomDebug(
            RoomType.Monster,
            MapPointType.Unassigned,
            encounter);
        return new CmdResult(
            task,
            success: true,
            $"Jumped to deterministic encounter: '{encounter.Id.Entry}' (floor={floor})");
    }

    /// <summary>
    /// 为首个参数补全全部已注册的遭遇 ID。
    /// </summary>
    /// <param name="player">请求补全的玩家。</param>
    /// <param name="args">当前已经输入的命令参数。</param>
    /// <returns>遭遇 ID 候选或后续参数上下文。</returns>
    public override CompletionResult GetArgumentCompletions(Player? player, string[] args)
    {
        if (args.Length <= 1)
        {
            var candidates = ModelDb.AllEncounters.Select(encounter => encounter.Id.Entry).ToList();
            return CompleteArgument(candidates, Array.Empty<string>(), args.FirstOrDefault() ?? "");
        }
        return new CompletionResult
        {
            Type = CompletionType.Argument,
            ArgumentContext = CmdName
        };
    }
}
