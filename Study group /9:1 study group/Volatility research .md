# Volatility research 

- HV是用[過去股價波動](https://slashtraders.com/tw/blog/thinkorswim-backtesting-options/)的標準差計算的波動率 ( RV: realized ) 

- IV是個向future 看的預測波動指標。





### IV 

> IV 通常藉由 BS model 計算

**隱含波動率（IV）**：反映市場對未來股票價格波動性的期望，從期權價格中推出。

**高隱含波動率**：表示市場預期價格可能出現大幅波動，導致期權價格更高。

**低隱含波動率**：表示市場預期價格波動較小，期權價格較低。

**市場情緒的反映**：IV幫助交易員了解市場情緒，做出更明智的交易決策。





### VIX ( Volaility index  )

- 「S&P500 指數未來 30 天的隱含波動率 」

- 通常VIX恐慌指數的數值在10-20之間，如果碰到戰爭、災害、恐怖事件等重大的突發事件時，恐慌指數會急速上升，這時候美股市場往往也會下跌。

   

   | **恐慌指數** | **狀態** | 
   |---|---|
   | 低於20 | 投資人心裡安定，市場處於穩定。 | 
   | 接近20 | 投資人警惕性升高，市場可能即將出現波動。 | 
   | 在20\~30 | 投資人心裡出現恐慌，市場出現波動。 | 
   | 高於30 | 投資人恐慌心裡加劇，市場波動增強。 | 
   





## More volatility Index 

- ADX 

   - lagging indicator 

- ATR 

   - pure volatility measurement 

   - Market. overall sentiment 

      - 盤整 —> 趨勢 ( or reverse ) 

   - lagging ( 一個 bar 結束才會知道 high low open close ) 

- BB 

   - think of it as an volatility indicator 

   - 怎麼魔改？

- BB squeeze ( + 肯特納 ) 



### Vega 

![Screenshot 2024-08-26 at 11.29.17 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-26%20at%2011.29.17 AM.png)

- Volatility is not equal to Vega 

- Vega is the sensitivity of a future or option to the “ implied volatility “ 



![Screenshot 2024-08-26 at 11.30.50 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-26%20at%2011.30.50 AM.png)



### Greek letter ：風險係數 

![Screenshot 2024-08-26 at 11.32.49 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-26%20at%2011.32.49 AM.png)













---

- 我們將這10天價格變化的標準差計算出來為2.54%。

   因為每年約有252個交易日，所以我們把2.54%的標準差修正為年的標準差40.3%，就等於HV。

   ![Pasted 2024-08-26-11-18-29.png](./Volatility%20research%20-assets/Pasted%202024-08-26-11-18-29.png)

   

這個方法用來將短期（例如10天）的標準差擴展為年度標準差，背後的理論基礎來自於**隨機過程的標準差縮放規則**，特別是布朗運動（Brownian Motion）模型。

會成立是基於：

- 市場效率假說 EMH 

- Random Walk Theory 



### Method : 

**假設獨立性和正態分佈**：

- 這種方法假設每天的回報是相互獨立的，並且遵循相同的分佈（通常假設為正態分佈）。這意味著短期標準差可以通過縮放來推導出長期標準差。

**平方根法則**：

- 根據隨機過程的理論，如果回報的標準差為 σ\\sigmaσ，那麼 TTT 天的回報標準差應該為 σ×T\\sigma \\times \\sqrt{T}σ×T​。這就是為什麼我們可以用短期標準差乘以時間跨度的平方根來得到長期標準差。

**年化標準差的重要性**：

- 年化標準差（通常稱為歷史波動率，Historical Volatility, HV）是投資者和風險管理者用來衡量資產價格在一年內波動幅度的重要指標。這使得年化波動率成為比較不同資產波動性時的一個標準化指標。



Application : 

- 期權定價模型

- VAR模型





### Delta 

- 標的(underlying future ) 變化1, 選擇權價格的變動率 

- 看漲期權：( 1, 100 ) 

- 看跌期權：( -1, -100 ) 

- calculation : Code 







### PCE index  : 衡量 通脹指標

**PCE 指數與 CPI 指數的差異**

- **範圍更廣**：PCE 指數不僅調查家庭購買情況，還包括企業銷售和非個人自付支出（如保險理賠）。

- **權重更新頻率更快**：PCE 指數的權重更新頻率比 CPI 指數快，能更準確地反映消費者行為的變化，因此 FED 更傾向於使用 PCE 作為調整利率政策的參考指標。

- **核心 PCE 指數**：扣除掉波動較大的能源和食物價格，更能反映通貨膨脹的基本趨勢。

- 這是目前 FED 用來判斷是否繼續升息的重要經濟指標。



---

### 波動率( vix option ) 與台指期 論文 

- 成交量的變化會迅速反應資訊對金融市場的影響

- 價量關係論文 ：

   - clarl 混合分配假說

- 價格波動與成交量存「正向」關係 

- K bars 型態： dempaster and Jones 

   - 頭肩型

   - 通道 

- Zero sum games 

   - 展望理論 （前景理論）：

      - behavioral economics 

      - 每個人基於初始狀況的不同（參考點不同），對風險會有不同的態度 

      - 人不再是基於「理性」的假設，而是加入對賺賠、發生機率高低等條件的不對稱心理作用

   - 討價還價 

      - full info 輪流出價的討價還價模型 

      - 合作博弈 

   - 演化與博弈論 

      - 「頻率依賴」的選擇行為 

         - 頻率依賴：一個性狀或行為在群體中的效果如何，取決於其他成員有多少人擁有相同的性狀或行為 （除了受自然環境影響，也受到該性狀在群體中頻率的影響） 

         - 例如，如果某一行為在群體中非常普遍，這個行為可能會變得不再那麼有利，因為競爭會變得激烈或資源會變得稀缺。

         - 相反，如果某行為在群體中非常少見，它可能會給持有這種行為的個體帶來額外的優勢，因為競爭較少或資源更豐富。

      - 因此，種群中特定表行的適應度，依賴於他們在群體中的頻率分佈 

      - 頻率依賴適應度：

         當大多數人都合作時，那個體採合作的適應度較高，然而當合作過於普遍時，採取自私或掠奪性行為的個體可能更有優勢，因為他們可以利用他人的合作而不用付出這麼多代價。

         #### 這邊或許可以plot 出一個曲線，也就是個體在一個閾值之下，合作都是比較好的。超過了之後，背叛的優勢就會比較高（非常賽局的東西）

   

### 波動率與價格的連動

- volatility 常根 fear index 聯想在一起 

   - 當股價下跌 ，波動度過高，市場參與者願意付出大量選擇權權利金來規避投資組合的風險 。因此，高波動率通常代表過度悲觀的市場預期 

   - 不過，過度悲觀，就代表做多的好機會。因此，極端波動度指標會被當成一種逆勢操作的訊號

- HV, Garch, stochastic volatility.  vs. IV 

   - IV 通常被相信有比較好的預測力 （基於 option 算出來）



### 台指期產業分類 

> 想法：
>
> - 半導體佔、金融過半 
>
> - 對沖：做多半導個股，做空台指期 （相對個股較便宜的保險）
>
>    - 點差：這裡會差到多少 （台指期應該會算好這件事，因為一開始用途就是對沖風險）
>
>    - 不能假設其他成份股不動，如果半導強勢，整體卻沒這麼高，代表市場對其他產業預期不高
>
>       

![Screenshot 2024-08-27 at 8.27.16 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%208.27.16 AM.png)



### 期現貨套利

1\. **期貨價格高於現貨價格（期貨溢價）**

**套利策略**：賣出期貨合約，買入現貨股票。

**作法**：

1. 當發現台指期的價格高於現貨指數時，可以在期貨市場賣出相應數量的台指期貨合約。

   1. 同時，在現貨市場買入相應價值的台股現貨股票，通常是買入指數成分股，以確保整體價值與期貨合約價值對應。

2. 等到期貨合約到期時，期貨價格和現貨價格趨於一致。你可以將期貨合約平倉，同時賣出現貨股票，從中獲利。

**獲利來源**：套利者利用期貨溢價進行操作，在合約到期時，期貨價格應該接近現貨價格，從而鎖定中間的價差為利潤。

2\. **期貨價格低於現貨價格（期貨折價）**

**套利策略**：買入期貨合約，賣出現貨股票。

**作法**：

1. 當台指期價格低於現貨指數時，可以在期貨市場買入相應數量的台指期貨合約。

2. 同時，在現貨市場賣出相應價值的台股現貨股票，通常是賣出指數成分股。

3. 到期後，當期貨價格和現貨價格趨於一致時，平倉期貨合約，同時回購現貨股票。

**獲利來源**：在期貨折價時，通過買入期貨合約和賣出現貨股票，套利者可以在期貨價格和現貨價格趨於一致時賺取差價。

3\. **考量因素**

**交易成本**：套利操作涉及同時在兩個市場進行交易，因此需要考慮交易成本，如手續費、交易稅等。這些成本可能會影響套利的利潤空間。

**市場流動性**：現貨市場和期貨市場的流動性可能會影響交易的執行速度和價格。在流動性低的情況下，可能難以按計劃價格進行交易。

**風險控制**：儘管套利通常被認為是低風險策略，但在極端市場情況下，如市場劇烈波動、合約到期時的意外情況等，仍可能存在風險。因此，投資者需要設置止損和風險管理措施。

4\. **執行時間**

**即時執行**：套利操作需要在發現點差異常時立即執行，以避免價格迅速回歸正常時錯失機會。

**持有時間**：通常套利交易會在合約到期前保持倉位，直到期貨和現貨價格趨於一致，這個過程可能需要數天至數周不等。





### VIX 

> 利用ｏｐｔｉｏｎ近月與次近🈷️所有價外合約的價格來計算，可用以反應選擇權市場對未來短期內股票市場波動程度的預期 

- 反應未來 30 days 波動程度

- \~ Normal Dist 

- e.g. IF 15, 代表預期年波動率為 15 % 

> 查：CBOE : 波動率指數編制公式





## 交易策略設定 

- 透過樣本內波動率指數和台指期close price 作「相關係數」分析

   - 要怎麼做？

   - 要哪些統計方式 ？

- 波動率對原始投資報酬的「有效」、「無效」進場點：

   - 有效：篩出拉的進場點使策略會剩餘未加入波動率指標（原始投資策略）

   - 無效：相反

- K bars setting : 

   - 5, 10, 15, …, 40 

- 原始投資策略：

   - close \[t\]  - close \[ t-1 \] \* 成交口數 = k 棒力道

   - 往前推進 \[ 10, 70\] 合計值產生多空力道累計值

   + \>0 , 多頭，<0 空

      透過最新一跟 k bar 結束後，持續更新多空力道累計值作「巡還交易」

      



### Result on original strategy 

![Screenshot 2024-08-27 at 9.46.50 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%209.46.50 AM.png)





![Screenshot 2024-08-27 at 9.47.18 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%209.47.18 AM.png)





## Volatility index vs. 台指期 index  

> 比較台指 close + daily return vs. VIX close + daily return 

![Screenshot 2024-08-27 at 9.49.50 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%209.49.50 AM.png)



### 期貨與波動率cum return 相關係數 

> 這邊波動度都是指「波動度指數」

![Screenshot 2024-08-27 at 9.50.51 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%209.50.51 AM.png)

- 相關係數為負： - 1 （圖中的 1 ) 

- 正： 0 

- insight ：大多數為負

   - meaning : 波動率cum return 與 期貨 cum return 呈負相關

   

![Screenshot 2024-08-27 at 9.54.39 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%209.54.39 AM.png)

- Insight: 看得累積時長（週期越長）越久，台指與波動反向的機率越高

- 

- 下圖：大致上呈現負相關，但不是所有時候

   - 可能下跌時，波動度會較高

   - 可能上漲時，波動度相對不會這麼高





## Adding volatility index into original strat 

### 做多

![Screenshot 2024-08-27 at 9.58.57 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%209.58.57 AM.png)



### 波動度 > 往前推N個 5 min 波動率 EMA 

![Screenshot 2024-08-27 at 10.02.15 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.02.15 AM.png)

- 看累積點數，加入波動度濾網，績效皆沒有明顯優化 

   ![Screenshot 2024-08-27 at 10.03.23 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.03.23 AM.png)

   ![Screenshot 2024-08-27 at 10.05.02 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.05.02 AM.png)

- 原始投資策略1487點不論往前1個五分鐘波動率移動平均數或往前8推個五分鐘波動率移動平均數，皆有不同表現

- 其中往前15分鐘波動率移動平均數(往前推3個)績效來到1780點，與原始投資策略相比多了293點。





### 做空 

![Screenshot 2024-08-27 at 10.07.16 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.07.16 AM.png)

- 在波動指數「小於」波動率指數 EMA 情況下，都無明顯優化 



![Screenshot 2024-08-27 at 10.08.42 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.08.42 AM.png)

- 「大於」情況下，除了往前7, 8 個5 min 波動率 ema 以外，皆有明顯成長 

- 原始投資策略342點不論是往前推1個五分鐘波動率移動平均數或是往前推6個五分鐘波動率移動平均數，整體績效皆會明顯優化

- 其中往前推5分鐘波動率移動平均數(往前推1個)績效來到1334點，與原策略342點相比，多了992點大幅提升整體策略績效。



> 小結：加入波動度濾網
>
> - 做多時，波動指數「小於」 波動指數 ema 有效
>
> - 做空時，波動指數「大於」 波動指數 ema 有效 



## 原始與加入波動度整體差異分析

![Screenshot 2024-08-27 at 10.13.10 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.13.10 AM.png)



![Screenshot 2024-08-27 at 10.13.20 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.13.20 AM.png)



![Screenshot 2024-08-27 at 10.13.39 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.13.39 AM.png)

![Screenshot 2024-08-27 at 10.14.31 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.14.31 AM.png)

- 優化和原始都大量擊敗大盤 

- 以上為累積點數 



### 交易筆數 

- 以後策略有多空，應該分開 analyze 

- 總交易次數下降，不管對於曝險或手續費皆為有利 

![Screenshot 2024-08-27 at 10.15.00 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.15.00 AM.png)





### 波動度分析

- 為什麼我們要隨機抽取幾天，然後算當日震幅？

![Screenshot 2024-08-27 at 10.17.16 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.17.16 AM.png)



### Volatility index’s high, low, close

- threshold : 20 —> 進入恐慌

![Screenshot 2024-08-27 at 10.18.14 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-27%20at%2010.18.14 AM.png)





## Conclusion 

- Zero sum game : 台指期

   - 有人買就有人賣，有人轉就有人賠

   - 以「成交量」當成策略主軸，算出成交量多空力道

- high volatiltiy —> better return 

- When VIX > 20 , 加入濾網將大幅提升整體交易策略 

- 與原始策略相比，如果波動度不夠大，那效果也不會大 





---

## Search  

- 大、小、微台的交易量比較 

- 零和遊戲概念在台指期上的應用：

   - 有人買進，代表一定有相對應得一個人賣出

- volatility smile 

- 厚尾現象 







### 價內、價外

> 一個合約，價內才有履約的意義

![Screenshot 2024-08-28 at 8.42.21 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-28%20at%208.42.21 AM.png)





## Sentimental Stress index paper

### Search 

- Bitcoin —> FinBert 

   - Finbert 可以直接做semtiment 

- credit spread 

   - 高風險與低風險債卷的收益的收益率差異 

   - 有點像是風險貼水

- CDS contract 

   - credit default swap 

   - 對借高風險客戶的一種保險：如果default 可以拿到補償

- market contagion

   - inside market chain reaction 

### Method of paper 

two alternatives : 

- stress index  

- stress index + news sentiment( 0, 1)

   - news signal: SMA 10 

      - z score to normalize 

   - stress index : 

      - using comprehensive risk analysis

### Computing Stresss index 

Indicators 

- Vix 

- Ted spread 

- CDS index 

- volatility data of major equity : stocks, bonds. commodities 

Steps

- z score normalization 

- *we aggregate these z-scores by their respective categories such as equities, emerging*

   *bonds, government bonds, financial stocks, foreign exchange, commodities, interest*

   *rates, and corporate credit to form category-specific stress indicators.*

- average it 

- scale it to fall btwn ( 0, 1) 



### Result 

- news signal is more sensitive to the stress index 

- the stress index alone performes the best besides the \[news + SI \] and \[ SI \] switch btwn, stands for its robustness 





---

## 風險平價模型

- risk parity 

- 注重資產間的風險關係 & 整體投資組合的風險特徵

   ### 原理

   - 風險貢獻均衡 

      - 傳統上可能是按照資金比例分配

      - 這邊會讓每一個分配所分到的「風險」是相等的

         - 意味風險本身較低的就會被分配到較多的部位？

   - 多樣化

      - 股票

      - 債券

      - commodoties 

   - 動態調整 

   ### 步驟

   - 衡量風險

      - std

      - volatility 

   -  資產corr 分析

      - 因為會牽涉到「組合」問題

   - 根據以上兩點，計算每種資產對「組合風險」的貢獻

      - 個別風險 vs. 組合風險 

   - 定期再平衡

   ### Data 

   - return 

   - volatility: std 

   - corr btwn asset 

      - 評估：

         - 分散化程度

         - 風險聚集 

   - market condition 

      - macro econ index 

      

## Model 

> goal : 建立一個portfolio, 目標是每個資產對總體風險的貢獻是平等的
>
> OPtimimzation 
>
> - **目标函数**：最小化资产风险贡献之间的差异，例如，通过最小化风险贡献的方差或其他度量差异的方法。
>
> - **约束条件**：包括权重之和等于1（ ∑(n，*i*=1) ​ *wi*​=1），以及权重非负等其他可能的投资限制。



### Risk contribution RC 

![Screenshot 2024-08-31 at 10.49.59 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-31%20at%2010.49.59 AM.png)



- w : weight , sigma( P) : 總體風險 

- 对于投资组合中的每个资产*i*，其风险贡献可以定义为该资产的权重*Wi*​ 乘以该资产与整个投资组合风险的 「边际贡献」。

- RC*i*​是资产*i*的风险贡献，*wi*​是资产*i*在投资组合中的权重

- *σ*(*P*)是投资组合*P*的总体风险（例如，标准差或波动率），∂*σ*(*P*)​/∂*wi*​是投资组合风险对资产*i*权重的偏导数，表示资产*i*权重变化对投资组合总体风险的影响。

- portfolio total risk （分子） 

   ![Screenshot 2024-08-31 at 10.52.23 AM.png](./Volatility%20research%20-assets/Screenshot%202024-08-31%20at%2010.52.23 AM.png)

   - Sigma: covariance matrix 





## 實際分配

- 实际的风险平价模型实现会涉及更复杂的数据分析和计算，包括:

   - 考虑预期收益

   - 动态调整相关性预测

   - 以及采用更高级的优化算法等。

   - 真实世界的资产配置还会考虑到:

      - 交易成本

      - 市场流动性

      - 税收因素

      - 投资者的具体风险偏好



## Common Strat 

以下是上述內容的重點整理：

1. **基於波動率的風險平價**：

   - 調整資產權重，使每個資產對總風險的貢獻相等。

   - 需要估計各資產的波動率和相關性。

2. **風險貢獻平價**：

   - 更深入地平衡每個資產對總風險的邊際貢獻。

      - 邊際貢獻 

   - 需要構建協方差矩陣並解決優化問題。

      - Covariance Matrix 建構 

3. **多因子風險平價**：

   - 考慮多個市場因子對資產收益的影響（如市場、規模、價值、動量等）。

   - 平衡這些因子的風險貢獻，以實現多樣化的風險分散。

4. **動態風險平價**：

   - 根據市場條件變化動態調整資產權重。

   - 基於市場波動性、經濟周期或其他宏觀經濟指標進行調整。

5. **槓桿風險平價**：

   - 使用槓桿增加對低風險資產的投資，以提高預期收益。

   - 常用於固定收益投資組合中，透過借貸購買更多債券。

6. **全球宏觀風險平價**：

   - 在全球範圍內分配資產，通過跨國界的資產類別實現風險平價。

   - 利用全球資產的多樣性降低組合的整體風險。



### 顯著特點 

- data driven 高度依賴 

- 多資產 適用性 

- 極端情況抗震性較好

   - 但如果整體黑天鵝，還是難逃 





## Code 

- Total portfolio volatility 

```python
portfolio_volatility 
= np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))  
# based on formula 

```





- Marginal RC 

![Screenshot 2024-08-31 at 1.08.34 PM.png](./Volatility%20research%20-assets/Screenshot%202024-08-31%20at%201.08.34 PM.png)

```python
marginal_risk_contributions = 
np.dot(covariance_matrix, weights) / portfolio_volatility

```

### 推導

>  這邊sigma(p) 對 w 偏微分，為什麼會產生const 2 把分母消掉？

![Screenshot 2024-08-31 at 1.13.33 PM.png](./Volatility%20research%20-assets/Screenshot%202024-08-31%20at%201.13.33 PM.png)





- RC : risk contributons 

   ![Screenshot 2024-08-31 at 1.36.21 PM.png](./Volatility%20research%20-assets/Screenshot%202024-08-31%20at%201.36.21 PM.png)

   ```python
   risk_contributions = 
   weights * marginal_risk_contributions  # based on formula 
   
   ```



- target risk contributions 

![Screenshot 2024-08-31 at 1.39.00 PM.png](./Volatility%20research%20-assets/Screenshot%202024-08-31%20at%201.39.00 PM.png)



- consraint function 

> type : equal 
>
> `fun: lambda x: np.sum(x) - 1` 是一個匿名函數（lambda 函數），用於計算權重的總和與1之間的差異。
>
> 具體來說：
>
> - `x` 是一個權重向量（例如，包含所有資產的權重）。
>
> - `np.sum(x)` 計算權重向量的總和，這應該是一個接近1的數值。
>
> - `np.sum(x) - 1` 計算總和與1之間的差異，這個差異應該等於零。

```python
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

```



- Optimization 

   ```python
   optimized_result = minimize(risk_parity_objective, initial_weights,
                               args=(covariance_matrix,), method='SLSQP',
                               constraints=constraints, bounds=bounds) 
   ```



- **注意**：這裡的逗號非常重要，因為它告訴 Python 這是一個包含單個元素的元組。如果省略逗號，Python 會將其解釋為一個單獨的變量，而不是元組。

- `SLSQP` 代表「Sequential Least SQuares Programming」，這是一種常用的約束優化算法，適合解決有邊界和約束條件的連續變量優化問題。





### Result 

> - 我們想要求出 分別 risk normalize 之後的 weight 
>
> - RC 應該相同 
>
> - weight 應該 volatility 越高而越小，但目前全部均等 
>
> 




