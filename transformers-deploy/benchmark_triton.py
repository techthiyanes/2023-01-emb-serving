#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse

import numpy as np
import tritonclient.http
import tqdm

#from transformer_deploy.benchmarks.utils import print_timings, setup_logging, track_infer_time

import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Union

import numpy as np
import torch


def print_timings(name: str, timings: List[float]) -> None:
    """
    Format and print inference latencies.

    :param name: inference engine name
    :param timings: latencies measured during the inference
    """
    mean_time = 1e3 * np.mean(timings)
    std_time = 1e3 * np.std(timings)
    min_time = 1e3 * np.min(timings)
    max_time = 1e3 * np.max(timings)
    median, percent_95_time, percent_99_time = 1e3 * np.percentile(timings, [50, 95, 99])
    print(
        f"[{name}] "
        f"mean={mean_time:.2f}ms, "
        f"sd={std_time:.2f}ms, "
        f"min={min_time:.2f}ms, "
        f"max={max_time:.2f}ms, "
        f"median={median:.2f}ms, "
        f"95p={percent_95_time:.2f}ms, "
        f"99p={percent_99_time:.2f}ms"
    )


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set the generic Python logger
    :param level: logger level
    """
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=level)


@contextmanager
def track_infer_time(buffer: List[int]) -> None:
    """
    A context manager to perform latency measures
    :param buffer: a List where to save latencies for each input
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    buffer.append(end - start)


def generate_input(
    seq_len: int, batch_size: int, input_names: List[str], device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Generate dummy inputs.
    :param seq_len: number of token per input.
    :param batch_size: first dimension of the tensor
    :param input_names: tensor input names to generate
    :param device: where to store tensors (Pytorch only). One of [cpu, cuda]
    :return: Pytorch tensors
    """
    assert device in ["cpu", "cuda"]
    shape = (batch_size, seq_len)
    inputs_pytorch: Dict[str, torch.Tensor] = {
        name: torch.ones(size=shape, dtype=torch.int32, device=device) for name in input_names
    }
    return inputs_pytorch


def generate_multiple_inputs(
    seq_len: int, batch_size: int, input_names: List[str], nb_inputs_to_gen: int, device: str
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate multiple random inputs.

    :param seq_len: sequence length to generate
    :param batch_size: number of sequences per batch to generate
    :param input_names: tensor input names to generate
    :param nb_inputs_to_gen: number of batches of sequences to generate
    :param device: one of [cpu, cuda]
    :return: generated sequences
    """
    all_inputs_pytorch: List[Dict[str, torch.Tensor]] = list()
    for _ in range(nb_inputs_to_gen):
        inputs_pytorch = generate_input(seq_len=seq_len, batch_size=batch_size, input_names=input_names, device=device)
        all_inputs_pytorch.append(inputs_pytorch)
    return all_inputs_pytorch


def to_numpy(tensors: List[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
    """
    Convert list of torch / numpy tensors to a numpy tensor
    :param tensors: list of torch / numpy tensors
    :return: numpy tensor
    """
    if isinstance(tensors[0], torch.Tensor):
        pytorch_output = [t.detach().cpu().numpy() for t in tensors]
    elif isinstance(tensors[0], np.ndarray):
        pytorch_output = tensors
    elif isinstance(tensors[0], (tuple, list)):
        pytorch_output = [to_numpy(t) for t in tensors]
    else:
        raise Exception(f"unknown tensor type: {type(tensors[0])}")
    return np.asarray(pytorch_output)


def compare_outputs(pytorch_output: np.ndarray, engine_output: np.ndarray) -> float:
    """
    Compare 2 model outputs by computing the mean of absolute value difference between them.

    :param pytorch_output: reference output
    :param engine_output: other engine output
    :return: difference between outputs as a single float
    """
    return np.mean(np.abs(pytorch_output - engine_output))


#############

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="require inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--length",  help="sequence length", choices=(16, 128), type=int, default=16)
    parser.add_argument("--model",  help="model type", choices=("onnx", "tensorrt"), default="onnx")
    args, _ = parser.parse_known_args()

    setup_logging()
    model_name = f"transformer_{args.model}_inference"
    url = "127.0.0.1:8000"
    model_version = "1"
    batch_size = 1

    if args.length == 128:
        # from https://venturebeat.com/2021/08/25/how-hugging-face-is-tackling-bias-in-nlp/, text used in the HF demo
        text = """Today, Hugging Face has expanded to become a robust NLP startup, 
        known primarily for making open-source software such as Transformers and Datasets, 
        used for building NLP systems. “The software Hugging Face develops can be used for 
        classification, question answering, translation, and many other NLP tasks,” Rush said. 
        Hugging Face also hosts a range of pretrained NLP models, on GitHub, that practitioners can download 
        and apply for their problems, Rush added."""  # noqa: W291
    else:
        text = "This live event is great. I will sign-up for Infinity."

    triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
    assert triton_client.is_model_ready(
        model_name=model_name, model_version=model_version
    ), f"model {model_name} not yet ready"

    model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

    query = tritonclient.http.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
    model_score = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)
    time_buffer = list()
    for _ in tqdm.trange(1000, desc="warmup"):
        query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
        _ = triton_client.infer(
            model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
        )
    for _ in tqdm.trange(1000, desc="inference"):
        with track_infer_time(time_buffer):
            query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
            response = triton_client.infer(
                model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
            )

    print_timings(name="triton transformers", timings=time_buffer)
    