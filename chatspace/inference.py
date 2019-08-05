"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import re
from typing import Dict, Generator, Iterable, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import ChatSpaceDataset
from .data.vocab import Vocab
from .model import ChatSpaceModel
from .resource import CONFIG_PATH, JIT_MODEL_PATH, MODEL_DICT_PATH, VOCAB_PATH


class ChatSpace:
    def __init__(
        self,
        model_path: str = None,
        config_path: str = CONFIG_PATH,
        vocab_path: str = VOCAB_PATH,
        device: str = "cpu",
        from_jit: bool = True,
    ):
        self.config = self._load_config(config_path)
        self.vocab = self._load_vocab(vocab_path)
        self.device = torch.device(device)

        if model_path is None:
            from_jit = self._is_jit_available() if from_jit else False
            model_path = JIT_MODEL_PATH if from_jit else MODEL_DICT_PATH

        self.model = self._load_model(model_path, self.device, from_jit=from_jit)
        self.model.eval()

    def space(self, texts: Union[List[str], str], batch_size: int = 64) -> Union[List[str], str]:
        """
        띄어쓰기 하려는 문장을 넣으면, 띄어쓰기를 수정한 문장을 만들어 줘요!
        전체 문장에 대한 inference가 끝나야 결과가 return 되기 때문에
        띄어쓰기가 되는 순서대로 iterative 하게 사용하고 싶다면 space_iter함수를 하용하세요!

        :param texts: 띄어쓰기를 하고자 하는 문장 또는 문장들
        :param batch_size: 기본으로 64가 설정되어 있지만, 원하는 크기로 조정할 수 있음
        :return: 띄어쓰기가 완료된 문장 또는 문장들
        """

        batch_texts = [texts] if isinstance(texts, str) else texts
        outputs = [output_text for output_text in self.space_iter(batch_texts, batch_size)]
        return outputs if len(outputs) > 1 else outputs[0]

    def space_iter(self, texts: List[str], batch_size: int = 64) -> Iterable[str]:
        """
        띄어쓰기 하려는 문장을 넣으면, 띄어쓰기를 수정한 문장을 iterative 하게 만들어 줘요!
        모든 띄어쓰기가 끝날 때 까지 기다리지 않아도 되니 for 문에서 사용할 수 있어요.

        내부적으로는 띄어쓰기 하려는 문장(들)을 넣으면 dataset 으로 변환하고
        model.forward에 넣을 수 있도록 token indexing 과 batching 작업을 진행합니다.

        :param texts: 띄어쓰기를 하고자 하는 문장 또는 문장들
        :param batch_size: 기본으로 64가 설정되어 있지만, 원하는 크기로 조정할 수 있음
        :return: 띄어쓰기가 완료된 문장 또는 문장
        :rtype collection.Iterable[str]
        """

        dataset = ChatSpaceDataset(self.config, texts, self.vocab)
        data_loader = DataLoader(dataset, batch_size, collate_fn=dataset.eval_collect_fn)

        for i, batch in enumerate(data_loader):
            batch_texts = texts[i * batch_size : i * batch_size + batch_size]
            for text in self._single_batch_inference(batch=batch, batch_texts=batch_texts):
                yield text

    def _single_batch_inference(
        self, batch: Dict[str, torch.Tensor], batch_texts: List[str]
    ) -> Generator[str, str, None]:
        """
        batch input 을 모델에 넣고, 예측된 띄어쓰기를 원본 텍스트에 반영하여
        띄어쓰기가 완료된 텍스트를 iterative 하게 생성 합니다!

        :param batch: 'input', 'length' 두 키를 갖는 batch input
            input은 char 를 encoding 한 [batch, seq_len] 크기의 torch.LongTensor
            length는 각 sequence 의 길이 정보를 갖고 있는 [batch] 크기의 torch.LongTensor
            length를 사용하는 이유는 dynamic LSTM을 사용하기 위해서 pack_padded_sequence 를 사용하기 때문임

        :param batch_texts: batch 에 들어간 실제 원본 문장들
        :return: 띄어쓰기가 완료된 문장
        :rtype collection.Iterable[str]
        """
        # model forward for chat-space nn.Module
        output = self.model.forward(batch["input"], batch["length"])

        # make probability into class index with argmax
        space_preds = output.argmax(dim=-1).cpu().tolist()

        for text, space_pred in zip(batch_texts, space_preds):
            # yield generated text (spaced text)
            yield self.generate_text(text, space_pred)

    def generate_text(self, text: str, space_pred: List[int]) -> str:
        """
        prediction 된 class index 를 실제 띄어쓰기로 generation 하는 부분

        :param text: 띄어쓰기가 옳바르지 않은 원본 문장
        :param space_pred: ChatSpaceModel.forward 에서 나온 결과를
        argmax(dim=-1)한 [batch, seq_len] 크기의 3-class torch.LongTensor
        0: PAD_TARGET, 1: NONE_SPACE_TARGET, 2: SPACE_TARGET
        :return: 띄어쓰기가 반영된 문장
        """
        generated_sentence = list()
        for i in range(len(text)):
            if space_pred[i] - 1 == 1:
                generated_sentence.append(text[i] + " ")
            else:
                generated_sentence.append(text[i])

        joined_chars = "".join(generated_sentence)
        return re.sub(r" {2,}", " ", joined_chars).strip()

    def _get_torch_version(self) -> int:
        """
        string 으로 되어있는 torch version 을 비교할 수 있도록 int로 변환

        :return: torch 버젼의 int version
        """
        version_string = "".join(re.findall(r"[0-9]+", torch.__version__))
        return int(version_string)

    def _is_jit_available(self) -> bool:
        """
        jit을 이용해서 모델 로딩이 가능한 pytorch 버전인지 체크하기

        :return: jit 모델 가능 여부 (bool)
        """
        return self._get_torch_version() >= 110

    def _load_model(self, model_path: str, device: torch.device, from_jit: bool) -> nn.Module:
        """
        저장된 ChatSpace 모델을 불러오는 함수

        :param model_path: 모델이 저장된 path
        :param device: 모델을 불러서 어떤 디바이스의 메모리에 올릴지
        :param from_jit: torch.jit.TracedModel 으로 저장된 모델을 불러올지
        아니면 state_dict 로 저장된 dictionary를 불러올지 설정
        :return: 로딩된 모델을 return
        """
        if from_jit:
            try:
                model = self._load_model_from_jit(model_path)
            except RuntimeError:
                print("Failed to load jit compiled model. Please set ChatSpace(as_jit=False)")
                model = self._load_model_from_dict(model_path, device)
        else:
            model = self._load_model_from_dict(model_path, device)
        return model.to(device)

    def _load_model_from_dict(self, model_path: str, device: torch.device) -> ChatSpaceModel:
        """
        torch.save(model.state_dict()) 로 저장된 state_dict 를 이용해 모델 로딩

        :param model_path: 모델 weight 가 저장되어 있는 위치
        :param device: 어떤 device 에 모델 weight 를 바로 위치시킬 지
        :return: weight 가 로딩된 ChatSpace 모델 (nn.Module)
        """
        print("Loading ChatSpace Model Weight")
        model = ChatSpaceModel(self.config)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        return model

    def _load_model_from_jit(self, model_path: str) -> Union[torch.jit.ScriptModule, nn.Module]:
        """
        torch.jit.save(traced_model) 로 저장된 jit compiled 모델 로딩

        :param model_path: 모델 파일 위치
        :return: jit traced ScriptModule
        """
        print("Loading JIT Compiled ChatSpace Model")
        model = torch.jit.load(model_path)
        return model

    def _load_vocab(self, vocab_path: str) -> Vocab:
        """
        저장된 vocab 을 로딩

        :param vocab_path: vocab 위치
        :return: 로딩된 vocab
        """
        with open(vocab_path) as f:
            vocab_tokens = [line.strip() for line in f]
        vocab = Vocab(tokens=vocab_tokens)
        self.config["vocab_size"] = len(vocab)
        return vocab

    def _load_config(self, config_path: str) -> dict:
        """
        저장된 config 을 로딩

        :param config_path: config 위치
        :return: 로딩된 config
        """
        with open(config_path) as f:
            return json.load(f)
