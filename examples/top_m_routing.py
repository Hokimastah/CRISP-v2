from crisp import ClusteredCRISPClassifier


clf = ClusteredCRISPClassifier(
    router_encoder="resnet18",
    n_clusters=5,
    cluster_top_m=2,
    retriever="numpy",
    max_exemplars_per_class=30,
    device="cpu",
)

clf.add_folder("dataset")
result = clf.predict("test_image.jpg", cluster_top_m=2)

print(result)
