import time

import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from chatspace import ChatSpace


@pytest.fixture()
def spacer():
    return ChatSpace()


@pytest.fixture()
def target_corpus():
    return read_file("tests/resource/target.txt")


def evaluation(spacer, target_corpus, eval_type):
    path = f"tests/resource/{eval_type}.txt"
    start_time = time.time()
    pred_lines = [line for line in spacer.space(read_file(path), batch_size=256)]
    duration = time.time() - start_time
    print(f"Duration: {duration} example/sec {len(pred_lines) / duration}")

    return get_metric(pred_lines, target_corpus, eval_type)


def test_easy(spacer, target_corpus):
    metric = evaluation(spacer, target_corpus, "easy")

    assert metric["acc"] >= 0.98
    assert metric["precision"] >= 0.97
    assert metric["recall"] >= 0.97
    assert metric["f1"] >= 0.97


def test_normal(spacer, target_corpus):
    metric = evaluation(spacer, target_corpus, "normal")

    assert metric["acc"] >= 0.97
    assert metric["precision"] >= 0.97
    assert metric["recall"] >= 0.94
    assert metric["f1"] >= 0.95


def test_hard(spacer, target_corpus):
    metric = evaluation(spacer, target_corpus, "hard")

    assert metric["acc"] >= 0.95
    assert metric["precision"] >= 0.97
    assert metric["recall"] >= 0.89
    assert metric["f1"] >= 0.93


def read_file(path):
    with open(path) as f:
        return [line.strip() for line in f]


def sent2spacing(sent):
    spacing = []
    for i in range(len(sent) - 1):
        if sent[i] != " ":
            if sent[i + 1] == " ":
                spacing.append(1)
            else:
                spacing.append(0)
    # print(len(spacing), len(sent.replace(" ", ""))-1)
    assert len(spacing) == len(sent.replace(" ", "")) - 1
    return spacing


def get_metric(input_corpus, target_corpus, eval_type="easy"):
    input_corpus.sort(key=lambda x: x.replace(" ", ""))
    target_corpus.sort(key=lambda x: x.replace(" ", ""))
    assert len(input_corpus) == len(target_corpus)

    input_spacing = [sent2spacing(sent) for sent in input_corpus]
    target_spacing = [sent2spacing(sent) for sent in target_corpus]
    input_spacing_all = sum(input_spacing, [])
    target_spacing_all = sum(target_spacing, [])
    # print(len(input_spacing_all), len(target_spacing_all))
    assert len(input_spacing_all) == len(target_spacing_all)

    accuracy = accuracy_score(target_spacing_all, input_spacing_all)
    precision = precision_score(target_spacing_all, input_spacing_all)
    recall = recall_score(target_spacing_all, input_spacing_all)
    f1score = f1_score(target_spacing_all, input_spacing_all)

    print(
        f"eval_code:{eval_type}",
        f"accuracy\t{accuracy}\n",
        f"precision\t{precision}\n"
        f"recall\t{recall}\n",
        f"f1 score\t{f1score}\n",
    )
    return {"acc": accuracy, "precision": precision, "recall": recall, "f1": f1score}
