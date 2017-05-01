import data_helpers

sentences = data_helpers.load_data_and_labels("./data/train.tsv")
print(sentences[1])