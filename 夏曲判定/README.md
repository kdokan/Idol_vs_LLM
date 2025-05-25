
# アイドルの歌詞内容からLLMを使って夏曲判定をする。

## 目次

- [やったこと・背景](##やったこと・背景)
    - [背景](#背景)
    - [やってみたこと](やってみたこと)
- [分析STEP](#分析STEP)
    - [3アイドルの歌詞をスクレイピングする。](#3アイドルの歌詞をスクレイピングする。)
    - [Embeddingsモデルで歌詞をベクトル化し、夏というワードのコサイン類似度を見る。](#Embeddingsモデルで歌詞をベクトル化し、夏というワードのコサイン類似度を見る。)
    - [LLM as a judge](#llm-as-a-judge)
    
## やったこと・背景

### 背景

- 社外ハンズオンに出た際に、文章をベクトル化(Embedding)してコサイン類似度でポジネガやラベル付与を判別するというのが自分の引き出しになかった。
- さらに、LLM as a judgeをすることで、

### やってみたこと


### やってみての所感




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

<pre><code>
# 全アイドルLLM as a Judge実行 -----------------------------------
for i in range(len(res_embedding)):
    Lyric = res_embedding.iloc[i]["Lyric"]

    model = "gpt-4o"
    prompt = f"""
    あなたは、歌詞の内容を元に「夏っぽいかどうか」を判定するスペシャリストです。
    次のルールに従って結果を出力してください。

    1. 次のようなJSON形式で出力してください：
    {{
    "score": 数値（0.0〜1.0）, 
    "reason": "夏っぽさに関する説明"
    }}

    2. "score" は夏らしさのスコアを 0〜1 の範囲でNumericに出力してください。
    0 は夏とまったく関係ない、1 は非常に夏らしいことを意味します。

    以下が歌詞です：

    {Lyric}
    """

    response = openai.chat.completions.create(
        model = model,
        messages = [{"role":"user","content":prompt}]
    )
    content = response.choices[0].message.content.strip()
    # コードブロックを外す処理
    if content.startswith("```") and content.endswith("```"):
        content = "\n".join(content.split("\n")[1:-1]).strip()
    result = json.loads(content)
    res_embedding.iloc[i, res_embedding.columns.get_loc("score")] = result["score"]
    res_embedding.iloc[i, res_embedding.columns.get_loc("reason")] = result["reason"]
</code></pre>

[https://qiita.com/t-hashiguchi/items/06222acd1643bc209b44](https://qiita.com/t-hashiguchi/items/06222acd1643bc209b44)
