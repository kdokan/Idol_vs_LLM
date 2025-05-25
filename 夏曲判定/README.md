
# アイドルの歌詞内容からLLMを使って夏曲判定をする。

## 目次

- [やったこと・背景](##やったこと・背景)
- [分析STEP](#分析STEP)
    - [3アイドルの歌詞をスクレイピングする。](#3アイドルの歌詞をスクレイピングする。)
    - [Embeddingsモデルで歌詞をベクトル化し、夏というワードのコサイン類似度を見る。](#Embeddingsモデルで歌詞をベクトル化し、夏というワードのコサイン類似度を見る。)
    - [LLM as a judge](#llm-as-a-judge)
    
## やったこと・背景


## 分析STEP

- 個人的に好きな3アイドル(わーすた、JamsCollection、≠ME)の歌詞をスクレイピングしてくる
- OpenAI APIでEmbeddingsモデルを使って歌詞をベクトル化し、夏というワードのベクトルとのコサイン類似度で夏曲の判定を行う。
- LLM as a judgeで夏曲かそうじゃないか？を判定する。

### 3アイドルの歌詞をスクレイピングする。
このコードを見れば、だれでもスクレイピングできる。

[https://qiita.com/q-tyl/items/0aee46e8ec68497e1700](https://qiita.com/q-tyl/items/0aee46e8ec68497e1700)

### Embeddingsモデルで歌詞をベクトル化し、夏というワードのコサイン類似度を見る。


ベクトル化するアプローチとして、word2vecとかもあるけどもこれは単語に対してしかできないので文章に対してできるのはLLMの強み

<pre><code>
#%%
# ①「夏」というワードをベクトル化 ------------
TARGET_WORD = "夏"
res_llm_summer = openai.embeddings.create(
                                model = 'text-embedding-ada-002'
                                , input = TARGET_WORD
                                )
vec_summer = [d.embedding for d in res_llm_summer.data][0]

# 
def Embdding_by_openai(Lyric, vec_summer):
    # ②判定したい歌詞をベクトル化
    res_llm_lyric = openai.embeddings.create(
                                    model = 'text-embedding-ada-002'
                                    , input = Lyric
                                    )
    vec_lyric = [d.embedding for d in res_llm_lyric.data][0]

    # コサイン類似度
    similarity = cosine_similarity([vec_lyric], [vec_summer])[0][0]
    return similarity

for i in range(len(datasets)):
    Lyric = datasets.iloc[i]["Lyric"]
    datasets.at[i, "Similarity"] = Embdding_by_openai(Lyric, vec_summer)
res_embedding = datasets.sort_values("Similarity", ascending=False)
</code></pre>

### LLM as a judge
[https://qiita.com/t-hashiguchi/items/06222acd1643bc209b44](https://qiita.com/t-hashiguchi/items/06222acd1643bc209b44)
