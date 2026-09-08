# UFNet frozen expert cache audit

Status: **PASS**

- Source commit: `5ece2c65ba184faccf6c8cdccdc03132427c464b`
- MC-dropout seed/trials: 20260908 / 30
- Feature dimensions: {'finger': 232, 'speech': 1024, 'smile': 42}
- Split counts: {'train': {'rows': 690, 'participants': 516}, 'validation': {'rows': 215, 'participants': 167}, 'test': {'rows': 197, 'participants': 162}}
- Expert architecture: all three released models are `ShallowANN` (`Linear -> MC Dropout -> Sigmoid`).
- Representation used by Feature Adapters: exact preprocessed expert input, because these experts contain no learned penultimate hidden layer.

## Expert metrics

     split modality  rows    AUROC    AUPRC
     train   finger   690 0.842384 0.810746
     train   speech   690 0.859844 0.849112
     train    smile   690 0.876668 0.835957
validation   finger   215 0.744544 0.596198
validation   speech   215 0.872456 0.837135
validation    smile   215 0.896296 0.832997
      test   finger   197 0.803808 0.686951
      test   speech   197 0.875171 0.839286
      test    smile   197 0.815549 0.694012

All cache arrays are finite, labels/IDs agree across modalities, and every row exactly matches the frozen paper manifest.
