# STAGE Dataset

**STAGE: A Benchmark for Knowledge Graph Construction, Question Answering, and In-Script Role-Playing over Movie Screenplays**

A bilingual (Chinese and English) benchmark for narrative understanding over movie screenplays.

## Dataset Overview

STAGE contains structured annotations for understanding complex narratives in both Chinese and English films. The dataset includes:

- **Chinese**: Chinese movie scripts with annotations (42 movies)
- **English**: English movie scripts with annotations (109 movies)

Each movie contains:
- `script.json`: Scene-by-scene segmented movie scripts
- `question_pairs_final.csv`: Multi-hop question-answer pairs
- `event_summarization_refined.json`: Event-level narrative summaries

## Data Format

### script.json
```json
[
  {
    "_id": 1,
    "title": "Scene Title",
    "subtitle": "Scene Subtitle",
    "content": "Scene content with dialogue and actions..."
  }
]
```

### question_pairs_final.csv
Contains question-answer pairs designed to test multi-hop reasoning over the narrative.

### event_summarization_refined.json
Event-level summaries capturing key narrative elements and their relationships.

## Publication Status

**🚧 Currently Under Review at ACL 2025**

This is a partial release containing the core annotations. The full dataset, including additional annotations and evaluation benchmarks, will be released upon paper acceptance.

## Access to Full Dataset

If you need access to the complete dataset before publication, please fill out our data request form:

**[Request Full Dataset Access](https://forms.gle/4ByHYYwe5MHuVZtG8)**

We will review requests and grant access for research purposes on a case-by-case basis.

## Citation

If you use this dataset in your research, please cite our paper (citation will be updated upon publication):

```bibtex
@article{tian2026stage,
  title={STAGE: A Benchmark for Knowledge Graph Construction, Question Answering, and In-Script Role-Playing over Movie Screenplays},
  author={Tian, Qiuyu and Li, Yiding and Chen, Fengyi and Liu, Zequn and Kong, Youyong and Guo, Fan and Li, Yuyao and Shen, Jinjing and Xie, Zhijing and Luo, Yiyun and others},
  journal={arXiv preprint arXiv:2601.08510},
  year={2026}
}
```

## Contact

For questions or issues regarding the dataset, please:
- Open an issue in this repository
- Fill out the [data request form](https://forms.gle/4ByHYYwe5MHuVZtG8) for access inquiries

---

**Note**: This is a preliminary release for research preview. More comprehensive documentation and analysis tools will be provided with the full release.