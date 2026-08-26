import random
from collections import Counter, OrderedDict
from pathlib import Path


TARGET_COUNTS = {
    "command": 55,
    "qa": 45,
    "fallback": 10,
    "short": 10,
}

SEED = 20260826

WAKE_FORMS = [
    "",
    "飒智，",
    "",
    "你好，",
    "",
    "",
]

USER_EXAMPLES = {
    "command": [
        "小飒小飒，帮我拿瓶可乐",
        "小飒，带我去厕所",
        "小飒，开始讲解",
        "小飒，停一下",
        "小飒，带我去主会场",
        "小飒小飒，显示地图",
        "小飒，声音大一点",
        "小飒小飒，返回充电",
        "小飒，帮我拿瓶水",
        "小飒，向右转",
        "向左转",
        "给我打招呼",
    ],
    "qa": [
        "小飒小飒，介绍飒智智能科技有限公司",
        "小飒小飒，介绍Sage Robot One",
        "小飒，WAIC是什么",
        "小飒，介绍Sage Dog",
        "小飒，介绍你们公司的产品",
        "小飒小飒，你能做什么",
        "小飒，你叫什么名字",
        "小飒，Sage Robot One有什么功能",
        "小飒小飒，厕所在哪里",
        "小飒，飒智智能科技有限公司是做什么的",
        "你们公司是做什么的",
        "介绍你们公司的产品",
        "你叫什么名字",
        "介绍Sage Dog",
    ],
}


def normalize_text(text):
    return text.strip().replace("“", "").replace("”", "").replace(" ，", "，")


def add(rows, category, text):
    text = normalize_text(text)
    if text:
        rows.setdefault(category, OrderedDict())[text] = None


def choose_prefix(index, force_wake=False):
    if force_wake:
        return "飒智，"
    return WAKE_FORMS[index % len(WAKE_FORMS)]


def add_variants(rows, category, intents, templates, limit=None):
    count = 0
    for intent_index, intent in enumerate(intents):
        for template_index, template in enumerate(templates):
            prefix = choose_prefix(intent_index + template_index)
            add(rows, category, f"{prefix}{template.format(**intent)}")
            count += 1
            if limit and count >= limit:
                return


def build_command(rows):
    for text in USER_EXAMPLES["command"]:
        add(rows, "command", text)

    destinations = [
        "厕所",
        "洗手间",
        "主会场",
        "服务台",
        "签到处",
        "出口",
        "入口",
        "展台",
        "创新展区",
        "媒体中心",
        "嘉宾休息室",
        "餐饮区",
        "休息区",
        "会议室",
        "问询处",
        "充电区",
        "A区",
        "B区",
        "Sage Robot One 展台",
        "Sage Dog 展台",
        "WAIC 主论坛",
        "WAIC 展区",
    ]
    nav_templates = [
        "带我去{place}",
        "请带我去{place}",
        "帮我导航到{place}",
        "我要去{place}",
        "{place}怎么走",
        "领我去{place}",
        "送我到{place}",
        "能不能带我去{place}",
        "去{place}",
    ]
    add_variants(rows, "command", [{"place": place} for place in destinations], nav_templates, limit=30)

    movement = [
        "向左转",
        "向右转",
        "左转一下",
        "右转一下",
        "往前走",
        "向前走两步",
        "往后退一点",
        "停一下",
        "马上停下",
        "原地等待",
        "靠左一点",
        "靠右一点",
        "走慢一点",
        "走快一点",
        "跟着我",
        "不要跟着我",
        "回到原点",
        "返回充电",
    ]
    for idx, action in enumerate(movement[:10]):
        add(rows, "command", f"{choose_prefix(idx)}{action}")
        add(rows, "command", f"{choose_prefix(idx + 2)}请{action}")

    items = [
        "可乐",
        "矿泉水",
        "咖啡",
        "纸巾",
        "宣传册",
        "资料袋",
        "参会证",
        "麦克风",
        "笔",
        "水杯",
    ]
    delivery_templates = [
        "帮我拿瓶{item}",
        "帮我拿一瓶{item}",
        "给我拿{item}",
        "递给我{item}",
        "帮我取一下{item}",
        "把{item}送到服务台",
        "把{item}送到主会场",
    ]
    add_variants(rows, "command", [{"item": item} for item in items], delivery_templates, limit=15)

    interaction_actions = [
        "开始讲解",
        "停止讲解",
        "继续讲解",
        "暂停讲解",
        "重新讲一遍",
        "打开问答模式",
        "退出问答模式",
        "打开导航模式",
        "退出导航模式",
        "打开讲解模式",
        "退出讲解模式",
        "显示日程",
        "显示地图",
        "播放视频",
        "暂停视频",
        "继续播放",
        "关闭视频",
        "声音大一点",
        "声音小一点",
        "说慢一点",
        "给我打招呼",
        "向大家问好",
        "欢迎一下来宾",
        "开始巡航",
        "结束巡航",
        "取消当前任务",
        "确认执行",
        "不要执行",
        "面对观众",
        "转向屏幕",
        "开始接待下一位观众",
        "把屏幕切到产品介绍",
    ]
    for idx, action in enumerate(interaction_actions):
        add(rows, "command", f"{choose_prefix(idx)}{action}")
        if idx % 3 == 0:
            add(rows, "command", f"{choose_prefix(idx + 1)}麻烦你{action}")


def build_qa(rows):
    for text in USER_EXAMPLES["qa"]:
        add(rows, "qa", text)

    hotword_questions = [
        "介绍WAIC",
        "介绍一下WAIC",
        "WAIC是什么",
        "WAIC今天有什么活动",
        "WAIC今天有哪些论坛",
        "WAIC今天有哪些嘉宾",
        "WAIC的会议日程是什么",
        "WAIC主论坛在哪里",
        "WAIC展区怎么走",
        "帮我查询WAIC日程",
        "帮我查一下WAIC活动时间",
        "介绍Sage Robot One",
        "介绍一下Sage Robot One",
        "Sage Robot One是什么",
        "Sage Robot One有什么功能",
        "Sage Robot One能做什么",
        "Sage Robot One能导航吗",
        "Sage Robot One怎么交互",
        "介绍Sage Dog",
        "介绍一下Sage Dog",
        "Sage Dog是什么",
        "Sage Dog有什么功能",
        "Sage Dog能做什么",
        "Sage Dog是机器狗吗",
        "介绍飒智智能科技有限公司",
        "介绍一下飒智智能科技有限公司",
        "飒智智能科技有限公司是做什么的",
        "飒智智能科技有限公司有哪些产品",
        "飒智智能科技有限公司有什么技术",
    ]
    for idx, question in enumerate(hotword_questions):
        add(rows, "qa", f"{choose_prefix(idx, force_wake=idx % 2 == 0)}{question}")

    company_questions = [
        "你们公司是做什么的",
        "你们公司叫什么",
        "你们公司在哪里",
        "你们公司有哪些产品",
        "介绍你们公司的产品",
        "你们主要做哪类机器人",
        "你们的机器人能落地在哪些场景",
        "你们的核心技术是什么",
        "你们有哪些解决方案",
        "你们服务过哪些行业",
        "你们机器人安全吗",
        "你们机器人怎么避障",
        "你们机器人怎么定位",
        "你们机器人能坐电梯吗",
        "你们机器人能自动充电吗",
    ]
    for idx, question in enumerate(company_questions):
        add(rows, "qa", f"{choose_prefix(idx)}{question}")
        add(rows, "qa", f"{choose_prefix(idx + 3)}请问{question}")

    product_questions = [
        "Sage Robot One适合什么场景",
        "Sage Robot One和普通导览机器人有什么区别",
        "Sage Robot One可以做展馆讲解吗",
        "Sage Robot One能不能配送物品",
        "Sage Dog适合什么场景",
        "Sage Dog可以巡检吗",
        "Sage Dog能在复杂地面行走吗",
        "机器狗有什么优势",
        "导览机器人有什么优势",
        "服务机器人能解决什么问题",
        "机器人多久需要充一次电",
        "机器人一次能工作多久",
        "机器人支持语音交互吗",
        "机器人支持多语言吗",
        "机器人能识别人脸吗",
        "机器人能连接大模型吗",
    ]
    for idx, question in enumerate(product_questions):
        add(rows, "qa", f"{choose_prefix(idx + 1)}{question}")
        if idx % 2 == 0:
            add(rows, "qa", f"{choose_prefix(idx + 4)}帮我问一下，{question}")

    event_questions = [
        "今天有哪些论坛",
        "今天有哪些嘉宾",
        "今天的会议日程是什么",
        "主会场在哪里",
        "服务台在哪里",
        "签到处在哪里",
        "洗手间在哪里",
        "厕所在哪里",
        "出口在哪里",
        "餐饮区在哪里",
        "媒体中心在哪里",
        "附近有什么展台",
        "这个展台是做什么的",
        "现在有什么活动",
        "下一场活动几点开始",
        "主论坛什么时候开始",
        "我在哪里可以领取资料",
        "附近哪里可以休息",
    ]
    for idx, question in enumerate(event_questions):
        add(rows, "qa", f"{choose_prefix(idx + 2)}{question}")
        if idx % 3 == 1:
            add(rows, "qa", f"请问{question}")

    robot_identity = [
        "你叫什么名字",
        "你是谁",
        "你能做什么",
        "你会哪些功能",
        "你来自哪里",
        "你是小飒吗",
        "你是飒智的机器人吗",
        "我可以怎么称呼你",
        "你能帮我做什么",
        "你能不能回答问题",
        "你能不能带路",
        "你能不能介绍产品",
    ]
    for idx, question in enumerate(robot_identity):
        add(rows, "qa", f"{choose_prefix(idx)}{question}")


def build_fallback(rows):
    fallback = [
        "你听得到我说话吗",
        "你听清楚了吗",
        "我刚才说什么",
        "请再听一遍",
        "请重新识别",
        "请重新回答",
        "不是这个意思",
        "我不是这个意思",
        "你理解错了",
        "识别错了",
        "刚才那句错了",
        "换一种说法",
        "我想问别的问题",
        "取消刚才的问题",
        "忽略刚才的话",
        "没关系",
        "稍等一下",
        "等我一下",
        "可以了",
        "没事了",
        "谢谢",
        "谢谢小飒",
        "再见",
        "先不用了",
        "我重新说",
        "听不清就靠近一点",
        "这里太吵了",
        "请大声一点回答",
        "请说慢一点",
        "你能不能重复一遍",
        "这个答案不对",
        "不用回答这个问题",
        "先停一下",
        "我没有叫你",
        "刚才不是对你说的",
    ]
    for idx, text in enumerate(fallback):
        add(rows, "fallback", f"{choose_prefix(idx)}{text}")
        if idx % 2 == 0:
            add(rows, "fallback", text)


def build_short(rows):
    short_texts = [
        "向左转",
        "向右转",
        "左转",
        "右转",
        "前进",
        "后退",
        "停下",
        "停一下",
        "别动",
        "开始讲解",
        "停止讲解",
        "继续讲解",
        "暂停讲解",
        "打开问答",
        "退出问答",
        "打开导航",
        "退出导航",
        "显示地图",
        "显示日程",
        "播放视频",
        "关闭视频",
        "返回充电",
        "取消任务",
        "确认",
        "不要",
        "打招呼",
        "介绍公司",
        "介绍产品",
        "介绍Sage Robot One",
        "介绍Sage Dog",
        "介绍WAIC",
        "去厕所",
        "去洗手间",
        "去主会场",
        "去服务台",
        "去签到处",
        "去出口",
        "拿可乐",
        "拿水",
        "大声一点",
        "小声一点",
        "慢一点",
        "快一点",
        "小飒小飒",
        "小飒",
        "飒智",
        "你好小飒",
        "谢谢",
        "再见",
        "可以了",
    ]
    for text in short_texts:
        add(rows, "short", text)


def pad_to_targets(rows):
    pads = {
        "command": [
            "帮我看看前面有没有路",
            "请避开前面的人",
            "跟我保持一点距离",
            "转向屏幕",
            "面对观众",
            "回到展台旁边",
            "开始接待下一位观众",
            "把屏幕切到产品介绍",
        ],
        "qa": [
            "你现在电量多少",
            "你能不能自己回去充电",
            "你能在展馆里自己导航吗",
            "你能识别障碍物吗",
            "你能和人连续对话吗",
            "你支持哪些语言",
            "你可以介绍展会吗",
            "你可以带我参观吗",
        ],
        "fallback": [
            "我没听清你的回答",
            "你再确认一下",
            "这个问题先跳过",
            "我等会再问",
            "别继续了",
            "请安静一点",
        ],
        "short": [
            "问答",
            "导航",
            "讲解",
            "巡航",
            "充电",
            "可乐",
        ],
    }
    wrappers = [
        "{text}",
        "请{text}",
        "麻烦你{text}",
        "能不能{text}",
        "帮我{text}",
        "现在{text}",
    ]
    for category, target in TARGET_COUNTS.items():
        for phrase in pads[category]:
            for wrapper in wrappers:
                for prefix in WAKE_FORMS:
                    add(rows, category, f"{prefix}{wrapper.format(text=phrase)}")
                    if len(rows[category]) >= target:
                        break
                if len(rows[category]) >= target:
                    break
            if len(rows[category]) >= target:
                break


def write_commands(rows):
    output = []
    for category, target in TARGET_COUNTS.items():
        items = list(rows[category].keys())[:target]
        output.extend((category, text) for text in items)

    lines = [
        "# Format:",
        "# category<TAB>text",
        "# Lines beginning with # are ignored.",
        "# Generated by scripts/generate_command_catalog.py.",
        "",
    ]
    current_category = None
    for category, text in output:
        if current_category and current_category != category:
            lines.append("")
        current_category = category
        lines.append(f"{category}\t{text}")
    Path("commands.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    rows = {category: OrderedDict() for category in TARGET_COUNTS}
    build_command(rows)
    build_qa(rows)
    build_fallback(rows)
    build_short(rows)
    pad_to_targets(rows)

    for category, target in TARGET_COUNTS.items():
        if len(rows[category]) < target:
            raise RuntimeError(f"{category} has only {len(rows[category])} rows, need {target}")

    output = write_commands(rows)
    print(f"Wrote {len(output)} commands to commands.txt")
    print("category:", dict(Counter(category for category, _ in output)))
    for keyword in ["WAIC", "Sage Robot One", "Sage Dog", "飒智智能科技有限公司", "小飒", "飒智"]:
        print(f"{keyword}: {sum(keyword in text for _, text in output)}")


if __name__ == "__main__":
    random.seed(SEED)
    main()
