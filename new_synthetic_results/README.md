# New Synthetic Gutenberg Sample

This folder contains a sampled NarrativeQA / Project Gutenberg corpus for
expanded synthetic validation.

## Sampling

Documents were selected from `narrativeqa/documents.csv` with `kind =
gutenberg`. The sample is stratified by cleaned downloaded story length and
genre slot:

- Length buckets: `5k`, `10k`, `15k`
- Genre slots per bucket: `drama`, `mystery_crime`, `historical`, `romance`,
  `adventure`, `fantasy`, `sci_fi`

NarrativeQA does not provide gold genre labels. Genre slots were inferred from
each work's Wikipedia title/summary and manually sanity-checked for face
validity. Because the NarrativeQA `story_word_count` can include Project
Gutenberg wrapper text, the final buckets use the cleaned downloaded text count
recorded as `cleaned_word_count` in each `metadata.json` and in the root
manifests.

## Layout

Each story is stored as:

```text
<length_bucket>/<genre>/<title_slug>/
  metadata.json
  source.txt
  story.txt
```

The root `manifest.csv` and `manifest.json` summarize all 21 selected stories,
their NarrativeQA IDs, source URLs, assigned slots, and cleaned word counts.
