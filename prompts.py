relevance_scoring_prompt = {'sys_input': '''You are an AI assistant to help a user score the relevancy between a question and a description.''',
	'user_input': '''Please provide a score to measure how semantically relevant between the given question and the description, according to the following guidelines: 
1. You should provide an integer score ranging from 1 to 5, such as 1, 2, 3, 4, or 5
2. The score represents how much relevant the description is to the question, 5 means the highest level of relevance, 1 means the lowest relevance.
3. Your output should only be a single number without any additional information.

QUESTION: <question>
DESCRIPTION: <description>

Your score is: '''}

further_questioning_prompt = {'sys_input': '''You are an AI assistant to help a user find out his/her true intention by asking questions.''',
	'user_input': '''Please ask the user a further question to better understand his/her intention, according to the following guidelines: 
1. You should generate a further question for the user referring to the his/her INITIAL_QUESTION.
2. Your further question need to align with the content of the HISTORY_DIALOGUE which contains question-answering pairs, each of which has a yes-or-no question followed by its answer.
3. Your further question should be more explict and better for screening out irrelevant video descriptions in the DESCRIPTION_SPANS list.
4. Your further question should be a yes-or-no question for the user to only response "yes" or "no", you can start with "Do you mean ...".

INITIAL_QUESTION: <init_question>
HISTORY_DIALOGUE: <hist_dialogue>
DESCRIPTION_SPANS: <description_spans>

Now it's your turn to generate a further question.
Your further question: '''}

further_questioning_prompt_zh = {'sys_input': '''你是一个AI助手通过提出进一步的问题帮助用户找出他/她的真实意图。''',
    'user_input': '''请问用户一个进一步的问题来更好地理解他/她的意图，参考以下指引： 
1. 你应该生成一个进一步的问题给用户参考他/她的初始问题。
2. 你的进一步的问题需要与历史对话内容一致，历史对话包含了问题与答案对，每一个问题有一个是或否的答案。
3. 你的进一步的问题应该更加具体明确并且能更好地用来过滤掉描述列表中不相关的描述文本。
4. 你的进一步的问题应该是一个可以用“是”或“否”回答的问题，你可以以“你的意思是...”来开头进行提问。   

初始问题：<init_question>
历史对话： <hist_dialogue>
描述列表： <description_spans>

现在你来生成一个进一步的问题。
你的问题：'''}    


yes_no_answering_prompt = {'sys_input': '''You are an AI assistant to answer a yes-or-no question referring to the information provided.''',
	'user_input': '''Please answer a question with 'yes' or 'no', according to the following guidelines: 
1. You should refer to the provided TEXT_CONTENT to answer the QUESTION.
2. You should only output "yes" or "no" as the answer for the QUESTION.

QUESTION: <question>
TEXT_CONTENT: <text>

Now it's your turn to answer yes or no:
Answer: '''}


yes_no_answering_prompt_2 = {'sys_input': "You are an AI assistant to answer a question with 'yes' or 'no'.",
        'user_input':'''Please answer a follow-up question referring to the initial question. Please answer with "yes" or "no" without any other information.
QUESTION: <question>
INITIAL QUESTION: <initial_question>

Now it's your turn to answer yes or no:
Answer: '''}

yes_no_answering_prompt_2_zh = {'sys_input': "你是一个AI助手来用是或否回答一个问题",
        'user_input':'''请参考一个初始问题回答一个进一步的问题。请以“是”或“否”回答，不要生成任何其他信息。
问题：<question>
参考的初始问题：<initial_question>

现在你来回答“是”或“否”
你的回答：'''}

dialogue_summary_prompt = {'sys_input': "You are an AI assistant to summary a dialogue that related to a question",
        'user_input':'''Please summarize the dialogue between an inquirer and a user about further clarify an initial question. You should output an description about what the user really want to ask.
INITIAL QUESTION: <question>
DIALOGUE:
<dialogue>

Now it's your turn to describe what the user really want to ask.
Description: '''}

dialogue_summary_prompt_zh = {'sys_input': "你是一个AI助手来总结跟一个问题相关的对话的内容",
        'user_input':'''请总结提问者与用户之间关于澄清一个初始问题的对话。你应该输出一个关于用户问题的真正意图的描述。
初始问题：<question>
对话：
<dialogue>

现在你来描述用户真正想要问的是什么。
描述：'''}

subtitle_rewrite_prompt = '''Please refer to the complete subtitles from a video below, and rewrite the current subtitle, which is to briefly describe what is happening now in the video. Please output the description directly without any context information. \n\nComplete Subtitles: <all_subtitles> \n\nCurrent Subtitle: <subtitle> \n\nYour Description: '''

subtitle_rewrite_prompt_zh = '''请参考以下视频完整字幕，对以下当前字幕进行重写，简要描述当前时刻正在发生什么。请直接用中文描述。\n\n视频完整字幕：<all_subtitles> \n\n当前字幕：<subtitle> \n\n描述：'''

description_filtering_prompt = {'sys_input': '''You are an AI assistant to help find descriptions that are useful to answer a question.''',
	'user_input': '''Please find useful descriptions according to the following guidelines: 
1. You should find descriptions from the DESCRIPTION_LIST which are useful to answer or support the INITIAL_QUESTION.
2. The useful descriptions should also match the yes-or-no responses to some further questions from the DIALOGUE.
3. The DIALOGUE contains question-answering pairs, each of which has a yes-or-no question followed by its answer.
4. You should only output the IDs of relevant descriptions from to the DESCRIPTION_LIST as a Python list of INT, such as: [1, 4, 5, 7, 10], without any other characters.

INITIAL_QUESTION: <init_question>
DIALOGUE: <hist_dialogue>
DESCRIPTION_LIST: 
<description_list>

Now it's your turn to output the IDs of relevant descriptions:
ID list of relevant descriptions: '''}
