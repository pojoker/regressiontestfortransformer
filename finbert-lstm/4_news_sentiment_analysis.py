# 导入pandas库，用于数据处理和CSV文件操作
import pandas as pd 

def FinBERT_sentiment_score(heading):
    """
    使用预训练的FinBERT模型计算金融新闻的情绪分数
    返回值范围：-1到1，-1表示负面，1表示正面
    参数：heading - 可以是单个新闻字符串，也可以是新闻列表
    如果传入多条新闻，会计算所有新闻情绪分数的平均值
    """
    # 导入transformers库的相关组件
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    
    # 加载FinBERT模型的tokenizer（分词器），用于将文本转换为模型可理解的格式
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    
    # 加载预训练的FinBERT模型（专门为金融领域训练的BERT模型）
    finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    
    # 创建情绪分析pipeline，将模型和tokenizer组合成一个可直接使用的工具
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    
    # 对输入的新闻（或新闻列表）进行情绪分析
    # 如果heading是列表，会对每条新闻都进行分析，返回结果列表
    # 如果heading是字符串，返回单个结果字典
    result = nlp(heading)
    
    # 确保result是列表格式（统一处理）
    if not isinstance(result, list):
        result = [result]
    
    # 处理所有新闻的情绪分析结果，计算平均值
    sentiment_scores = []
    for res in result:
        # 将每条新闻的情绪标签和分数转换为统一的分数（-1到1之间）
        if res['label'] == "positive":
            # 正面情绪：返回置信度分数（0到1之间）
            sentiment_scores.append(res['score'])
        elif res['label'] == "neutral":
            # 中性情绪：返回0
            sentiment_scores.append(0)
        else:
            # 负面情绪：返回负的置信度分数（-1到0之间）
            sentiment_scores.append(0 - res['score'])
    
    # 计算所有新闻情绪分数的平均值
    average_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return average_score


def VADER_sentiment_score(heading):
    """
    使用VADER情绪分析工具计算情绪分数
    返回值范围：-1到1，-1表示负面，1表示正面
    参数：heading - 新闻文本字符串
    """
    # 导入nltk库和VADER情绪分析器
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # 下载VADER所需的词典数据（如果还没有下载的话）
    nltk.download('vader_lexicon')
    
    # 再次导入（重复导入，可以删除）
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # 创建VADER情绪分析器实例
    analyzer = SentimentIntensityAnalyzer()
    
    # 对新闻文本进行情绪分析，返回包含正负面、中性分数的字典
    # 结果格式：{'neg': 0.2, 'neu': 0.5, 'pos': 0.3, 'compound': 0.1}
    result = analyzer.polarity_scores(heading)
    
    # 判断哪种情绪得分最高（正面、负面或中性）
    if result['pos'] == max(result['neg'], result['neu'], result['pos']):
        # 如果是正面情绪得分最高，返回正面分数
        return result['pos']
    if result['neg'] == max(result['neg'], result['neu'], result['pos']):
        # 如果是负面情绪得分最高，返回负的负面分数（转为负数）
        return (0 - result['neg'])
    else:
        # 如果是中性情绪得分最高，返回0
        return 0

# ========== 主程序开始 ==========

# 从CSV文件读取新闻数据
# news_data.csv格式：第一列是日期，后续列是News 1, News 2, ..., News 10
news_df = pd.read_csv("news_data.csv")

# 创建一个空列表，用于存储每天的情绪分数
BERT_sentiment = []

# 遍历数据框中的每一行（每一天的新闻数据）
for i in range(len(news_df)):
    # 提取当前行的所有新闻列（从第2列开始，即索引1及之后）
    # iloc[i, 1:] 表示：第i行，从第1列（索引1）开始到最后一列
    # tolist() 将pandas Series转换为Python列表
    news_list = news_df.iloc[i, 1:].tolist()
    
    # 过滤掉值为'0'的新闻（这些可能是占位符，表示当天没有那么多条新闻）
    news_list = [i for i in news_list if i != '0']
    
    # 调用FinBERT函数对新闻列表进行情绪分析
    # 函数会分析所有新闻，并返回所有新闻情绪分数的平均值
    score_BERT = FinBERT_sentiment_score(news_list)
    
    # 将计算得到的情绪分数添加到列表中
    BERT_sentiment.append(score_BERT)

# 注释掉的调试代码：打印第129行的数据
# print(news_df.iloc[129])

# 将计算得到的情绪分数列表添加为数据框的新列
news_df['FinBERT score'] = BERT_sentiment

# 将包含情绪分数的新数据框保存为CSV文件
news_df.to_csv("sentiment.csv")