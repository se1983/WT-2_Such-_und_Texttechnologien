WT-2_Such-_und_Texttechnologien


# Maximum likelihood

- Welcher Part ist am häufigsten

$$
ml = \frac{n_k}{\Omega}
$$

# Naive Bayes

## Bayessche Formel


$$
p(\beta | \alpha) = \frac{p(\alpha|\beta) \cdot p(\beta)}{p(\alpha)}
$$

$$
\mbox{A-Posteriori} = \frac{\mbox{likelihood} \cdot \mbox{prior}}{\mbox{evidenz}}
$$

## Naive bayes - Verreinfacht
- (Verhältnis statt w-keit , da keine evidenz)
$$
{\displaystyle {\hat {y}}={\underset {k\in \{1,\dots ,K\}}{\operatorname {argmax} }}\ p(C_{k})\displaystyle \prod _{i=1}^{n}p(x_{i}\mid C_{k})}
$$


# Decision Tree Learning

## Entropy

$$ E(S) = - \sum\limits_{i=1}^{n} p_1 \cdot \log_2(p_i) $$

## Information

$$ I(S, A) = - \sum\limits_{i=1}^{n} \frac{|S_i|}{|S|} \cdot E(S_i) $$

## Information Gain

$$ \mbox{IG}(S,A) = E(S) - I(S,A)$$

## Intrinsic Information

$$ \mbox{IntI}(S,A) = - \sum\limits_i \frac{|S_i|}{|S|} \cdot \log_2\left(\frac{|S_i|}{|S|}\right) $$

## Gain Ratio

$$
GR(S,A) = \frac{\mbox{Gain}(S,A)}{\mbox{IntI}(S,A)}
$$


## Gini Index

$$ \mbox{Gini}(S) = 1- \sum\limits_i p_i^2$$
$$ \mbox{Gini}(S, A) = \sum\limits_i \frac{|s_i|}{|S|} \cdot \mbox{Gini}(S) $$


## Pruning

$$
B \leftarrow A \rightarrow C
$$

```python
prune(B, C) if (
    B.len()/A.len() * B.err() + C.len()/A.len() * C.err()
) > A.err()
```



# Nearest Neighbor

## Distanz

### Euklidisch
$$
d_E(X,Y) = \sqrt{\sum\limits_{i=1}^n (x_i-y_i)^2}
$$

### Manhattan

$$
d_M(X,Y) = \sum\limits_{i=1}^n |x_i-y_i|
$$

## Normalisierung der Werte

$$
\mbox{normalize}(\alpha) = \frac{\alpha-\mbox{min}}{\mbox{max}-\mbox{min}}
$$

# Accuracy


## Messaure the Accuracy

- **TP**: true positives
- **FP**: false positives
- **FN**: false negatives
- **TN**: true negatives
- **P** = TP + FN -- all positive instances
- **N** = FP + TN -- all negative instances

### Predictive Accuracy

$$
p = \frac{TP + TN}{P + N}
$$

### Standard error
$$
\sqrt{\frac{1-p}{N}}
$$

### Recall (TP-Rate)

$$
\frac{TP}{P}
$$

### FP-Rate

$$
\frac{FP}{N}
$$

### Precision

$$
\frac{TP}{TP+FP}
$$

### F1 Score

$$
2 ...
TODO
$$

# Clustering

## k-means

1. Choose a value of k.
2. Select k objects in an arbitrary fashion. Use these as the initial set of k
centroids.
3. Assign each of the objects to the cluster for which it is nearest to the
centroid.
4. Recalculate the centroids of the k clusters.
5. Repeat steps 3 and 4 until the centroids no longer move.


### Distance
$$
\mbox{dist}(p_i, c_a) = \sqrt{ (x_i - x_{cluster-a})^2 + (y_i + y_{cluster-a})^2  }
$$

# Page Rank
$$
r_{k+1}(p_i) = d \cdot \left( \sum_{pj \in B_{p_i}}   \frac{r_k(pj)}{|pj|}  + \sum_{pj,\,|pj|=0}  \frac{r_k(p_j)}{N} \right) + \frac{t}{N}
$$


- $|p_j|$ : Anzahl aller Links aus $p_j$
- $p_j \in B_{p_i}$  Alle Seiten die auf $p_i$ zeigen

## Abbruchbedingung

$$
\sum\limits_{p_i, i \in 0,1,\dots, N} \left| r_{k+1} (p_i) - r_k (p_i) \right| \leq \delta
$$

# TF IDF

## Term Frequency
$tf(d,t)$ -- Wie oft kommt Term $t$ in Dokument $d$ vor?

- Log Term Freq

```python
0 if tf(d,t) == 0 else log(tf(d,t), 10)
```

## Inversed Document Frequency

$$
idf(t) = log_{10} \frac{N}{df(t)}
$$

- $df(t)$ -- Anzahl der Dokumente die Term $t$ enthalten

