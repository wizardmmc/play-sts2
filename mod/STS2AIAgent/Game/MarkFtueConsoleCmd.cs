using System;
using System.Linq;
using MegaCrit.Sts2.Core.DevConsole;
using MegaCrit.Sts2.Core.DevConsole.ConsoleCommands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Saves;

namespace STS2AIAgent.Game;

/// <summary>
/// 仅用于开发的控制台命令，用于把首次用户体验弹窗标记为已读：
///   markftue ascension_singleplayer_ftue [更多名称]
/// NAscensionSingleplayerFtue 是普通覆盖层节点，不遵循模态窗口协议；它不提供
/// can_confirm、can_dismiss 或游戏动作，因此无头实例无法关闭。它通过
/// SaveManager.SeenPopup 检查 FtueCompleted，且不像 SeenFtue 那样受
/// EnableFtues 控制，所以即使关闭教程也可能出现。进入角色选择前，通过与
/// UI 关闭弹窗相同的 MarkFtueAsComplete 路径写入名称，可让 StartRunLobby
/// 跳过该覆盖层。该命令只用于隔离的测试存档初始化。
/// </summary>
public sealed class MarkFtueConsoleCmd : AbstractConsoleCmd
{
    public override string CmdName => "markftue";

    public override string Args => "<ftue_name> [more names]";

    public override string Description => "Marks FTUE popups as seen (dev-only; e.g. ascension_singleplayer_ftue).";

    public override bool IsNetworked => true;

    public override CmdResult Process(Player? issuingPlayer, string[] args)
    {
        if (args.Length == 0)
        {
            return new CmdResult(success: false,
                "Usage: markftue <ftue_name> [...]. Known gate names include"
                + " ascension_singleplayer_ftue / ascension_multiplayer_ftue"
                + " / combat_reward_ftue / accept_tutorials_ftue.");
        }

        var manager = SaveManager.Instance;
        var alreadySeen = new System.Collections.Generic.List<string>();
        var marked = new System.Collections.Generic.List<string>();
        foreach (var raw in args)
        {
            var name = raw.Trim();
            if (name.Length == 0)
            {
                continue;
            }
            // 使用门卫同款 SeenPopup 判定；SeenFtue 会在 enable_ftues=false 时
            // 被短路为恒真，从而把未读状态误报为已读。
            if (manager.SeenPopup(name))
            {
                alreadySeen.Add(name);
                continue;
            }
            manager.MarkFtueAsComplete(name);
            marked.Add(name);
        }

        var message = marked.Count > 0
            ? "Marked as seen: " + string.Join(", ", marked)
            : "Nothing to mark (already seen: "
              + (alreadySeen.Count > 0 ? string.Join(", ", alreadySeen) : "none") + ").";
        return new CmdResult(success: true, message);
    }
}
