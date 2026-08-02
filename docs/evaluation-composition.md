# What my evaluation set couldn't tell me

PaperTrail answers questions about a set of PDFs by retrieving relevant chunks and
generating an answer grounded in them. Retrieval is the step that decides whether any
of it works: if the right passage never reaches the context window, generation quality
is irrelevant.

So I built the evaluation harness before touching retrieval. Hand-labeled questions
over the NIST AI Risk Management Framework, each tagged with the chunk that actually
contains the answer, scored on Recall@3, Recall@6, and MRR. No LLM in the loop, just
deterministic matching against known-correct chunks.

That was the right instinct, and it still let me draw a conclusion that was wrong in a
way I could not see from the aggregate numbers.

## The headline

The harness started at 13 questions. A day later I expanded it to 30, then added
cross-encoder reranking: retrieve 40 candidates with the embedding model, rescore them
with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keep the top few.

On the 30-question set, Recall@3 went from 0.70 to 0.80 and MRR from 0.602 to 0.732.
A clean single-variable comparison, and the number I have been quoting.

## What I found when I went back

Months later, writing this up, I re-ran the current pipeline against only the original
13 questions, reranking on and off.

Recall@3 went from 0.69 down to 0.62. Recall@6 from 0.85 down to 0.69.

Not a smaller gain. The opposite sign.

On that 13-question subset the per-question movement is four demotions against three
promotions. (Across the full 30 the split is six promotions and five demotions, which is
the number reported in the repo README. Both are correct; they describe different sets.)

Reranking did exactly what the wide candidate net was designed to enable: q6 went from a
miss to rank 1, and q11 and q13 both jumped to rank 1. But it also demoted q3 from 1 to
7, q7 from 1 to 8, q9 from 2 to 5, and q4 from 3 to a miss.

## Not a resolution problem

My first instinct was that 13 questions was simply too few to resolve a 10-point effect.
That is a real concern and the arithmetic supports it: each question is worth 7.7 points,
and the standard error of a proportion near 0.75 at n=13 is about 0.12, larger than the
effect itself.

But that is not what happened. An underpowered set fails to see a signal. This set
reported the opposite sign, consistently, with specific questions moving in specific
directions. That is a different kind of failure and it does not get fixed by adding
questions at random.

## What I can and cannot say about why

What the data supports: the improvement is concentrated in the questions added later,
and the two groups are equally hard for first-stage retrieval.

| Group       | Recall@3 (vector only) | Recall@3 (+ reranking) |
| ----------- | ---------------------- | ---------------------- |
| Original 13 | 0.69                   | 0.62                   |
| Added 17    | 0.71                   | 0.94                   |

On the added 17, reranking also lifts Recall@6 to 1.00. Four of the five regressions
across the full 30 sit inside the original 13. The aggregate 0.70 to 0.80 rests entirely
on the later questions.

And because vector-only performance is nearly identical across the two groups (0.69
against 0.71, with matching rank-1 rates), the later questions are not simply easier to
retrieve. Something about how the cross-encoder responds to them differs.

What I would like to say, and cannot: that the original 13 skewed toward definitional
Core-function lookups where the query terms appear verbatim in the target passage,
leaving a reranker no headroom and full exposure to its own domain mismatch. That is a
plausible mechanism and it fits most of the pattern.

It does not fit all of it. q6 asks what the MANAGE function entails, which is squarely a
Core-function lookup, and reranking rescued it from a miss to rank 1. If the mechanism
were clean, that question should have been among the casualties.

With five regressions across thirty questions, this data cannot separate the
explanations. I am leaving it unresolved rather than picking the tidy one, because
picking the tidy one is the same move that produced the original mistake: an explanation
that fit the aggregate and was never checked against the parts.

One more possibility I cannot rule out: the later questions were written by someone who
had already watched this pipeline fail, and that could shape phrasing in ways that
flatter a reranker without making the questions easier to retrieve. The vector-only
numbers rule out "easier." They do not rule out "phrased in a style this cross-encoder
happens to handle well."

What I will commit to is the weaker, better-supported claim. The evaluation set I wrote
first was not representative of the queries this intervention affects, and nothing in the
aggregate score would have told me that.

## The second problem, which is worse

`fetch_k=40` was not an arbitrary choice. I picked it after a diagnostic showed the
expanded set's misses sitting at first-stage ranks 12, 13, 19, and 33. A 20-candidate net
could not reach them; a 40-candidate net could.

That is a hyperparameter selected against the same 30 questions I then reported scores
on.

When the demotion pattern above suggested pinning strong first-stage hits, I swept that
too. `pin_n=2` beats the current pipeline on both recall metrics, 0.87 against 0.80 at
Recall@3 and 0.93 against 0.87 at Recall@6, recovering q3, q7, q9, and q18 while keeping
every reranker promotion inside the top 3.

That is two hyperparameters now tuned on the set I score against. I built an evaluation
harness specifically so that changes would be measured rather than guessed, and then
quietly turned it into something I was fitting to. It took writing this down to notice.

One reading note on that sweep, because it is a trap: MRR drops as `pin_n` rises, and
that is mechanical rather than a quality signal. Pinning reserves the first `pin_n`
slots, so a promoted passage can land at best at rank `pin_n + 1`. At `pin_n=2` all six
promoted questions sit at exactly rank 3. They did not get worse. They hit a ceiling I
imposed. MRR is the wrong instrument for reading a constraint you built on purpose.

## What I decided

The pipeline stays at `fetch_k=40` with reranking on. It wins on both recall metrics
across the full set, and the regression is now characterized instead of hidden, which is
a better position than reverting to a configuration I have no audit of.

Pinning does not ship. It is the crude version of the right idea, and adopting the best
of four configurations chosen on the test set would compound a problem I have already
made once. Reciprocal rank fusion blends the two orderings by rank instead of reserving
slots, so a passage both stages like can still reach rank 1. That is the experiment worth
running, and it is documented as the next step rather than run in a hurry to make this
writeup tidier.

## What I would do differently

Not "use more questions." The useful version is harder: before measuring an intervention,
ask what kinds of queries it could plausibly help, and check whether the evaluation set
contains any of them. My 13 questions were a reasonable test of whether retrieval worked.
They were a nearly worthless test of whether reranking helped, and nothing about the
aggregate score said so.

And when a hyperparameter gets chosen by looking at the evaluation set, that fact belongs
next to the number it produced, permanently, not in a footnote discovered later.

---

Code, evaluation harness, and full per-question results:
[PaperTrail](https://github.com/cqaxo/papertrail)
