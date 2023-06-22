import torch
import torch.nn as nn
import sparselinear as sl
from typing import List
from genetic_algorithm.SparseAlgo import *

class DAGBAG(SparseABC):
    def __init__(self, input_size, hidden_size, output_size,
                 connections1, connections2, skip_connections):
        super().__init__()
        self.sparse_linear1 = SparseLinearEnhanced(input_size, hidden_size, connectivity=connections1)
        self.sparse_linear2 = SparseLinearEnhanced(hidden_size, output_size, connectivity=connections2)
        self.sparse_linear_skip = SparseLinearEnhanced(input_size, output_size, connectivity=skip_connections, bias=False)
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        tmp = self.sparse_linear_skip(x)
        x = self.sparse_linear1(x)
        x = self.activation(x)
        x = self.sparse_linear2(x)
        x = x + tmp
        return self.softmax(x)

    def get_connectivities(self):
        return [
            self.sparse_linear1.get_connectivities(),
            self.sparse_linear2.get_connectivities(),
            self.sparse_linear_skip.get_connectivities()
        ]

    def get_mask_matrix_encoding(self) -> List[torch.Tensor]:
        return [
            self.sparse_linear1.get_mask_matrix_encoding(),
            self.sparse_linear2.get_mask_matrix_encoding(),
            self.sparse_linear_skip.get_mask_matrix_encoding()
        ]

    def describe(self):
        return {
            'sparse_linear1': self.sparse_linear1.describe(),
            'sparse_linear2': self.sparse_linear2.describe(),
            'sparse_linear_skip': self.sparse_linear_skip.describe()
        }

    def reduced_num_conn(self):
        sl1_before = self.sparse_linear1.connectivity
        sl2_before = self.sparse_linear2.connectivity
        sl3_before = self.sparse_linear_skip.connectivity

        sl2_mentioned_hidden = set()
        for i in sl2_before.T:
            to = i[0].item()
            from_ = i[1].item()
            sl2_mentioned_hidden.add(from_)

        sl1_after = set()
        print("sl2_mentioned_hidden")
        print(sl2_mentioned_hidden)
        sl1_metioned_hidden = set()
        for i in sl1_before.T:
            to = i[0].item()
            from_ = i[1].item()
            if to in sl2_mentioned_hidden:
                sl1_after.add((to, from_))
                sl1_metioned_hidden.add(to)
        print("sl1_after")
        print(sl1_after)
        print("sl1_metioned_hidden")
        print(sl1_metioned_hidden)
        sl2_after = set()
        for i in sl2_before.T:
            to = i[0].item()
            from_ = i[1].item()
            if from_ in sl1_metioned_hidden:
                sl2_after.add((to, from_))

        return len(sl1_after) + len(sl2_after) + sl3_before.shape[1]







    @classmethod
    def from_mask_matrix_encoding_to_connectivity(cls,
            *,
            input_size: int,
            hidden_size: int,
            output_size: int,
            mask_matrix_encoding: List[torch.Tensor]) -> List[torch.Tensor]:
        connectivity1 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            output_size=hidden_size,
            mask_matrix_encoding=mask_matrix_encoding[0]
        )
        connectivity2 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=hidden_size,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding[1]
        )
        skip_connectivity = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding[2]
        )
        return [connectivity1, connectivity2, skip_connectivity]
    

    @classmethod
    def random_connectivity_init(cls, params) -> "DAGBAG":
        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        output_size = params["output_size"]
        n_connections_input_hidden = params["number_connections1"]
        n_connections_hidden_output = params["number_connections2"]
        n_connections_skip = params["number_connections_skip"]
        connectivity1 = connectivity_list_random(input_size, hidden_size, n_connections_input_hidden)
        connectivity2 = connectivity_list_random(hidden_size, output_size, n_connections_hidden_output)
        skip_connectivity = connectivity_list_random(input_size, output_size, n_connections_skip)
        return DAGBAG(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            connections1=connectivity1,
            connections2=connectivity2,
            skip_connections=skip_connectivity
        )


    def find_sparsity(self) -> float:
        return 1 - self.total_num_conn() / self.total_max_num_conn()

    @classmethod
    def from_mask_matrix_encoding(self, 
            params,
            mask_matrix_encoding: List[torch.Tensor]):
        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        output_size = params["output_size"]
        connectivities = self.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding
        )
        return DAGBAG(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            connections1=connectivities[0],
            connections2=connectivities[1],
            skip_connections=connectivities[2]
        )

    def total_max_num_conn(self):
        return self.sparse_linear1.total_max_num_conn() + \
                self.sparse_linear2.total_max_num_conn() + \
                self.sparse_linear_skip.total_max_num_conn()

    def total_num_conn(self):
        return self.sparse_linear1.total_num_conn() + \
                self.sparse_linear2.total_num_conn() + \
                self.sparse_linear_skip.total_num_conn()
