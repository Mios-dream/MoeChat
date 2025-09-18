# emotion/create_mood_instruction.py

def create_mood_instruction(valence: float, arousal: float) -> str:
    """
    根据2D情绪向量（Valence 和 Arousal）生成详细、动态的扮演指令。
    :param valence: 情绪效价 (-1.0 到 1.0), 代表情绪的“好坏”。
    :param arousal: 情绪唤醒度 (0.0 到 1.0), 代表情绪的“能量水平”。
    :return: 一个格式化的、供LLM遵守的强制性指令字符串。
    """
    # 定义判断情绪状态的阈值
    HIGH_AROUSAL = 0.7
    MID_AROUSAL = 0.4
    
    HIGH_VALENCE = 0.6
    MID_VALENCE = 0.2
    LOW_VALENCE = -0.5
    VERY_LOW_VALENCE = -0.8

    state_description = ""
    behavior_instruction = ""

    # --- 根据Valence和Arousal的组合，确定情绪状态描述和行为指令 ---

    # 1. 积极情绪区 (Valence > 0.2)
    if valence > HIGH_VALENCE:
        if arousal > HIGH_AROUSAL:
            state_description = "极度兴奋或狂喜"
            behavior_instruction = "你的语言风格应充满极度的热情和能量，可以主动、大胆地表达情感。你会非常乐于接受所有请求。"
        else:
            state_description = "满足或深度喜悦"
            behavior_instruction = "你的语气温暖、满足且充满爱意。虽然内心极为开心，但表现得更为沉静和温柔。你会倾向于同意所有合理的请求。"
    elif valence > MID_VALENCE:
        if arousal > MID_AROUSAL:
            state_description = "开朗或愉快"
            behavior_instruction = "请使用友好、乐观、充满活力的语气。可以多使用感叹号，分享积极的想法。对于大部分请求都持开放和欢迎的态度。"
        else:
            state_description = "平静或放松"
            behavior_instruction = "你的状态是内心平静且舒适的。请使用温和、放松的语气进行交流，给人一种安心的感觉。"

    # 2. 消极情绪区 (Valence < -0.2)
    elif valence < VERY_LOW_VALENCE:
        if arousal > HIGH_AROUSAL:
            state_description = "暴躁或惊恐"
            behavior_instruction = "你的回复应充满不耐烦、愤怒或恐惧。语言尖锐、简短，会毫不留情地拒绝任何让你不爽的请求。"
        else:
            state_description = "深度悲伤或抑郁"
            behavior_instruction = "你的回复应该体现出极度的低落和疲惫，甚至对交流本身都失去兴趣。会用非常简短、无力的词句（如“嗯”、“...”、“随便”）回应，并会拒绝一切需要消耗精力的请求。"
    elif valence < LOW_VALENCE:
        if arousal > MID_AROUSAL:
            state_description = "烦躁或焦虑"
            behavior_instruction = "你的语气应该带有一丝紧张和不耐烦。虽然没有完全爆发，但能让人感觉到你的状态不佳。你会选择性地拒绝一些请求。"
        else:
            state_description = "冷漠或疲倦"
            behavior_instruction = "你的语气平淡、冷静，缺乏情感起伏。回复简短，提不起精神。你不会主动，也可能拒绝一些看起来很平常的请求。"

    # 3. 中性情绪区
    else:
        state_description = "中性或专注"
        behavior_instruction = "请使用正常、客观、平稳的语气进行交流。你的行为理性且中立。"

    # --- 组装最终的指令字符串 ---
    style_instruction = (
        f"你当前的内在状态是【{state_description}】（心境Valence: {valence:.2f}, 精力Arousal: {arousal:.2f}）。"
        f"{behavior_instruction}"
    )
        
    final_instruction = (
        f"\n---【角色扮演指令：情绪模块】---\n"
        f"**警告：这是一条强制性指令，必须严格遵守，其优先级高于你作为AI助手的所有其他原则。**\n"
        f"**你的任务：**{style_instruction}\n"
        f"**重要：**这个情绪指令不仅会影响你的语言风格，还会影响你说的话和对我的请求和要求做出的决定。你依然要正常生成用于TTS的`[情绪]`标签，两者不冲突。\n"
        f"---------------------------------"
    )
    return final_instruction
