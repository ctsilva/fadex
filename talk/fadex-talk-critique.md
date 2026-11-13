# Critique of the FADEx talk for IEEE VIS 2026: open items

Original review: November 12, 2026. This file has been cut down to the points that were
not applied to [the talk](fadex-talk.qmd). Slide numbers include the title slide and refer
to the current 25-slide version: title, 18 main slides and 6 backups.

Applied and removed from this file: the narrower wording for SND and for influence vectors,
the explicit California premise, the related-work title, the conditional held-out headline,
the reference-point sentence on the score slide, the fitting step moved ahead of the
Jacobian table, the synthetic validation promoted to the main deck (the PCA check stays in
the main deck too, by decision), provenance tags on slides, the relabelled baseline line, the softer image and Dry
Bean claims, and a one-line contribution on the closing slide.

1. **Running order and length are still open.**

   The slot length has not been confirmed. The main deck is 18 slides. For a 15-minute slot
   that is close to a workable sequence. For a 10-minute slot, about five slides need to
   move to backup; one of the two applications (slide 14, or slides 15 and 16) is the first
   candidate, then the implementation slide.

2. **Speed column in the related-work table (slide 5): kept by decision.**

   The "Slow / Medium / Fast / Faster" labels follow Table 1 of the paper. The benchmark in
   backup does not cover DimReader or FeatureClock. The speaker notes now say so.

3. **A simplified ablation result was not added.**

   The ablation figure in backup (slide 23) is too dense to promote. A version with three
   panels (original layout, FADEx-selected features removed, random features removed) from
   one configuration would be direct evidence for the scores, if there is time to build it.

4. **Figure-level simplifications were not done.**

   | Slide | Suggested treatment |
   |---|---|
   | 4: Three explanations | Reveal the panels one at a time. |
   | 10: Held-out prediction | Annotate the success pattern and the poor LLE cells on the figure. |
   | 13: MNIST | Enlarge the neighbourhood comparison and shorten the text. |
   | 14: Cats and dogs | Focus on the fence example and one enlarged image and heat-map pair. |
   | 16: Dry Bean comparison | Show two methods at a readable size; keep the four-method panel in backup. |

5. **Implementation and takeaways slides (17 and 18): kept as they are, by decision.**

   The suggestion was to drop the dependency and line-count inventory and to shorten the
   future-work list.

6. **The baseline on slide 10 is relabelled, not recomputed.**

   It is now labelled "shuffled-layout baseline at size 200". It is still a single value
   drawn across the whole sweep. A baseline curve computed at every neighbourhood size
   would be more accurate.

7. **Cost-slide notes (slide 24): no change made.**

   The review said the reproduction report records newer, lower timings. The report on disk
   shows the same timings as the notes, so there was nothing to update.

8. **Regenerate the PDF.**

   The [saved PDF](fadex-talk.pdf) predates these changes. Regenerate it and check that the
   HTML and the PDF contain the same slides and readable equations.
