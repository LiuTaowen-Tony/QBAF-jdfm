import torch
import torch.nn as nn
import sparselinear as sl
from typing import List
from sparselinear import SparseLinear
from abc import abstractmethod
import math
from genetic_algorithm.SparseAlgo import *

# def connectivity_list_random(in_size: int, out_size: int, num_connections: int):
#     col = torch.randint(low=0, high=in_size, size=(num_connections,)).view(1, -1).long()
#     row = torch.randint(low=0, high=out_size, size=(num_connections,)).view(1, -1).long()
#     connections = torch.cat((row, col), dim=0)
#     return connections

# def mask_matrix_2_connectivity_list(adjacency_matrix: torch.Tensor):
#     adjacency_matrix = adjacency_matrix.round()
#     adjacency_matrix = adjacency_matrix.type(torch.long)
#     assert adjacency_matrix.unique().tolist() in [[0], [1], [0, 1]]
#     adjacency_list = [[], []]
#     for i in range(adjacency_matrix.shape[0]):
#         row_non_zero = torch.nonzero(adjacency_matrix[i]).squeeze(1).tolist()
#         for j in row_non_zero:
#             adjacency_list[1].append(i)
#             adjacency_list[0].append(j)
#     return torch.tensor(adjacency_list)

# def adjacency_list_2_mask_matrix(adjacency_list: torch.Tensor, in_size: int, out_size: int):
#     # convert type to int
#     adjacency_list = adjacency_list.type(torch.int)
#     adjacency_matrix = torch.zeros(in_size, out_size, dtype=torch.int)
#     for row, col in (adjacency_list.T):
#         assert 0 <= col < in_size
#         assert 0 <= row < out_size
#         adjacency_matrix[col, row] = 1
#     return adjacency_matrix


# class SparseABC(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     @abstractmethod
#     def get_connectivities(self):
#         pass

#     @abstractmethod
#     def get_mask_matrix_encoding(self) -> List[torch.Tensor]:
#         pass

#     @abstractmethod
#     def from_mask_matrix_encoding_to_connectivity(self):
#         pass

#     def find_sparsity(self) -> float:
#         return 1 - self.total_num_conn() / self.total_max_num_conn()

#     @abstractmethod
#     def from_mask_matrix_encoding(self):
#         pass

#     @abstractmethod
#     def total_max_num_conn(self):
#         pass

#     @abstractmethod
#     def total_num_conn(self):
#         pass

    

# class SparseLinearEnhanced(SparseLinear, SparseABC):
#     def __init__(
#             self,
#             in_features: int, 
#             out_features: int, 
#             connectivity: torch.Tensor
#         ):
#         """
#         :param connectivity: connectivity matrix of shape (in_features, out_features) 
#         """
#         try:
#             super().__init__(
#                 in_features=in_features, 
#                 out_features=out_features, 
#                 connectivity=connectivity.to(torch.long))
#         except:
#             connectivity = torch.tensor([[0],[0]], dtype=torch.long)
#             super().__init__(
#                 in_features=in_features,
#                 out_features=out_features,
#                 connectivity=connectivity.to(torch.long))
#         self.mask_matrix = adjacency_list_2_mask_matrix(connectivity, in_features, out_features)

#     def total_max_num_conn(self):
#         return self.in_features * self.out_features
    
#     def total_num_conn(self):
#         return self.connectivity.shape[1]

#     def get_connectivities(self):
#         return self.connectivity

#     def get_mask_matrix_encoding(self):
#         return adjacency_list_2_mask_matrix(
#             self.connectivity,
#             self.in_features,
#             self.out_features)
    
#     @classmethod
#     def from_mask_matrix_encoding_to_connectivity(
#             cls, 
#             *, 
#             input_size:int,
#             output_size:int,
#             mask_matrix_encoding:torch.Tensor)->torch.Tensor:
#         return mask_matrix_2_connectivity_list(
#             mask_matrix_encoding.reshape(input_size, output_size)
#         )

#     @classmethod
#     def from_mask_matrix_encoding(
#         cls,
#         *,
#         input_size:int,
#         output_size:int,
#         mask_matrix_encoding:torch.Tensor
#     ) -> "SparseLinearEnhanced":
#        return SparseLinearEnhanced(
#            in_features=input_size,
#            out_features=output_size,
#            connectivity=mask_matrix_encoding
#        ) 

# def test_sparse_linear_enhanced():
#     in_size = 10
#     out_size = 20
#     num_connections = 30
#     connectivity = connectivity_list_random(in_size, out_size, num_connections)
#     sparse_linear = SparseLinearEnhanced(
#         in_features=in_size, 
#         out_features=out_size, 
#         connectivity=connectivity)
#     assert sparse_linear.find_sparsity() == (1 - num_connections / (in_size * out_size))
#     assert sparse_linear.mask_matrix.shape == (in_size, out_size)
#     assert (sparse_linear.mask_matrix == adjacency_list_2_mask_matrix(connectivity, in_size, out_size)).all()
#     assert (sparse_linear.mask_matrix == sparse_linear.get_mask_matrix_encoding()).all()
#     assert (sparse_linear.connectivity == sparse_linear.get_connectivites()).all()
#     assert (sparse_linear.connectivity == mask_matrix_2_connectivity_list(sparse_linear.mask_matrix)).all()
#     assert (sparse_linear.connectivity == sparse_linear.from_mask_matrix_encoding_to_connectivity(
#         input_size=in_size, output_size=out_size, mask_matrix_encoding=sparse_linear.get_mask_matrix_encoding())).all()
#     assert (sparse_linear.from_mask_matrix_encoding(
#         input_size=in_size, output_size=out_size, mask_matrix_encoding=sparse_linear.get_mask_matrix_encoding()).connectivity == sparse_linear.connectivity).all()

# class SparseProd(SparseABC):
#     def __init__(self, *, 
#         connectivity: torch.Tensor, 
#         input_size: int, 
#         joint_feature_size: int):
#         super().__init__()
#         """
#         :param connectivity: connectivity matrix of shape (input_size, joint_feature_size) 
#         """
#         self.input_size = input_size
#         self.joint_feature_size = joint_feature_size
#         self.connectivity = connectivity
#         self.mask_matrix = adjacency_list_2_mask_matrix(connectivity, input_size, joint_feature_size)
#         self.mask_matrix.requires_grad_(False)

#     @classmethod
#     def from_mask_matrix_encoding(cls,  input_size: int, joint_feature_size: int, mask_matrix_encoding: torch.Tensor):
#         connectivity = mask_matrix_2_connectivity_list(
#             mask_matrix_encoding.reshape(input_size, joint_feature_size)
#             # input_size=input_size,
#             # joint_feature_size=joint_feature_size,
#             # mask_matrix_encoding=mask_matrix_encoding
#         )
#         return cls(
#             connectivity=connectivity, 
#             input_size=input_size, 
#             joint_feature_size=joint_feature_size)

#     @classmethod
#     def from_mask_matrix_encoding_to_connectivity(
#             cls, 
#             *,
#             input_size:int,
#             joint_feature_size:int,
#             mask_matrix_encoding:torch.Tensor) -> torch.Tensor:
#         return mask_matrix_2_connectivity_list(
#             mask_matrix_encoding.reshape(input_size, joint_feature_size)
#         ) 
    
#     def total_max_num_conn(self):
#         return self.input_size * self.joint_feature_size
    
#     def total_num_conn(self):
#         return self.connectivity.shape[1]

#     def get_connectivities(self):
#         return self.connectivity

#     def get_mask_matrix_encoding(self) -> torch.Tensor:
#         return self.mask_matrix.flatten()

#     def forward(self, x: torch.Tensor):
#         bsize_x, input_size_x = x.shape
#         assert self.input_size == input_size_x

#         # TODO : bug here, x[x==0] = 1 ... original value can == 1
#         x = x.unsqueeze(-1) 
#         assert x.shape == (bsize_x, self.input_size, 1)
#         # add joint_feature_size dimension
#         # x size = (bsize, input_size, 1)
#         x = x * self.mask_matrix 
#         x = x.clone()
#         x[x == 0] = 1
#         # print(x)
#         assert x.shape == (bsize_x, self.input_size, self.joint_feature_size)
#         # broadcast over joint_feature_size
#         # mask out non-connected weights
#         # x size = (bsize, input_size, joint_feature_size)

#         x = x.prod(dim=1, ) 
#         x = x.clone()
#         x[x == 1] = 0
#         # product over input_size
#         assert x.shape == (bsize_x, self.joint_feature_size)
        
#         return x

# def test_sparse_prod():
#     in_size = 4
#     joint_feature_size = 5
#     num_connections = 3
#     connectivity = torch.Tensor([[1,2,2], [0,1,3]])
#     sparse_prod = SparseProd(
#         connectivity=connectivity, 
#         input_size=in_size, 
#         joint_feature_size=joint_feature_size)
#     assert sparse_prod.find_sparsity() == (1 - num_connections / (in_size * joint_feature_size))
#     assert sparse_prod.mask_matrix.shape == (in_size, joint_feature_size)
#     assert (sparse_prod.mask_matrix == adjacency_list_2_mask_matrix(connectivity, in_size, joint_feature_size)).all()
#     assert (sparse_prod.mask_matrix == sparse_prod.get_mask_matrix_encoding().reshape(in_size, joint_feature_size)).all()
#     assert (sparse_prod.connectivity == sparse_prod.get_connectivities()).all()
#     assert (sparse_prod.connectivity == mask_matrix_2_connectivity_list(sparse_prod.mask_matrix)).all()
#     assert (sparse_prod.connectivity == sparse_prod.from_mask_matrix_encoding_to_connectivity(
#         input_size=in_size, joint_feature_size=joint_feature_size, mask_matrix_encoding=sparse_prod.get_mask_matrix_encoding())).all()
#     assert (sparse_prod.from_mask_matrix_encoding(
#         input_size=in_size, joint_feature_size=joint_feature_size, mask_matrix_encoding=sparse_prod.get_mask_matrix_encoding()).connectivity == sparse_prod.connectivity).all()
#     # sparse_prod.connectivity = torch.Tensor([[1, 2], [3, 4]])

#     print(sparse_prod.get_connectivities())
#     print(sparse_prod.get_mask_matrix_encoding())
#     x = torch.Tensor([[1, 2, 3, 4], [11,12,13,14]])
#     # print(sparse_prod(x))

#     assert (sparse_prod(x) == torch.torch.Tensor([[  0.,   0.,   8.,   0.,   0.], [  0.,  11., 168.,   0.,   0.]])).all()

class ProdCollectiveAS(SparseABC):
    def __init__(self, *, input_size: int, output_size: int, connectivities: torch.Tensor):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.connectivities = connectivities
        _, self.n_connections = connectivities.shape
        self.mask_matrix = adjacency_list_2_mask_matrix(connectivities, input_size, output_size).T.to(torch.float32).unsqueeze(0).contiguous()
        self.weight = nn.Parameter(torch.rand(size=(self.output_size,)))

    @classmethod
    def from_mask_matrix_encoding(cls, *,  input_size: int, output_size: int, mask_matrix_encoding: torch.Tensor):
        connectivity = ProdCollectiveAS.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding)
        return cls(
            connectivity=connectivity, 
            input_size=input_size, 
            joint_feature_size=output_size)

    @classmethod
    def from_mask_matrix_encoding_to_connectivity(
            cls, 
            *,
            input_size:int,
            output_size:int,
            mask_matrix_encoding:torch.Tensor) -> torch.Tensor:
        return mask_matrix_2_connectivity_list(
            mask_matrix_encoding.reshape(input_size, output_size)
        ) 
    
    def total_max_num_conn(self):
        return self.input_size * self.output_size
    
    def total_num_conn(self):
        return self.output_size

    def get_connectivities(self):
        return self.connectivities

    def get_mask_matrix_encoding(self) -> torch.Tensor:
        return self.mask_matrix.squeeze(0).T.flatten().contiguous()

    def describe(self, linear:SparseLinearEnhanced):
        connectivity = self.get_connectivities()
        result = []
        for i in connectivity.T:
            to = i[0].item()
            from_ = i[1].item()
            weight = self.weight[to].item()
            bias = linear.bias.tolist()[to].item()
            result.append(
                {
                    "from": from_,
                    "to": to,
                    "bias": bias,
                    "weight": weight
                }
            )
        return result

    def forward(self, x: torch.Tensor):
        # impl 1
        x = torch.log(x)
        x = x.clamp(-50, 50)
        x = x.unsqueeze(-1)
        x = self.mask_matrix @ x
        x = x.squeeze(-1)
        x = torch.exp(x)
        return x * self.weight

        # bsize_x, input_size_x = x.shape
        # assert self.input_size == input_size_x
        # result = torch.zeros(bsize_x, self.output_size)
        # x = x.repeat(1, 1, self.output_size)
        # assert x.shape == (bsize_x, self.input_size, self.output_size)
        # x[:, ~self.mask_matrix] = 1
        # x = x.prod(dim=1)
        # self.weight[self.weight_mask] = 0
        # return x * self.weight

        self.mask_matrix = self.mask_matrix.to(torch.bool)
        assert self.mask_matrix.shape == (self.input_size, self.output_size)
        for ith_batch, batch in enumerate(x):
            for ith_result, ith_mask in enumerate(self.mask_matrix.T):
                result[ith_batch, ith_result] = batch[ith_mask].prod() * self.weight[ith_result]

        return result

def test_prod_collective_as():
    input_size = 3
    output_size = 2
    connectivities = torch.Tensor([[1, 0, 0], [1, 1, 0]])
    prod_collective_as = ProdCollectiveAS(
        input_size=input_size, 
        output_size=output_size, 
        connectivities=connectivities)
    prod_collective_as.weight = nn.Parameter(torch.tensor([1., 1., 1.]))
    x = torch.arange(30, dtype=torch.float32).view(6,5)
    # assert (prod_collective_as(x) == torch.Tensor([[  0.,  17.], [  0., 168.]])).all()
    assert prod_collective_as.total_num_conn() == 3
    assert prod_collective_as.total_max_num_conn() == 6
    # assert list(map(list, prod_collective_as.get_connectivities())) == [connectivity_input_joint, connectivity_joint_output]
    assert (prod_collective_as.get_mask_matrix_encoding() == adjacency_list_2_mask_matrix(
        adjacency_list=connectivities, 
        in_size=input_size, 
        out_size=output_size).flatten()).all()




# class ProdCollectiveAS(SparseABC):
#     def __init__(self, *, input_size, joint_feature_size, output_size, connections_input_joint, connections_joint_output):
#         super().__init__()
#         self.input_size = input_size
#         self.joint_feature_size = joint_feature_size
#         self.output_size = output_size
#         self.prod = SparseProd(
#             connectivity=connections_input_joint, 
#             input_size=input_size, 
#             joint_feature_size=joint_feature_size, 
#         )
#         self.sparse_linear = SparseLinearEnhanced(joint_feature_size, output_size, connectivity=connections_joint_output)

    
#     def total_max_num_conn(self):
#         return self.input_size * self.joint_feature_size + self.joint_feature_size * self.output_size
    
#     def total_num_conn(self):
#         return self.prod.total_num_conn() + self.sparse_linear.total_num_conn()

#     @classmethod
#     def from_flattened_mask_matrices(
#             cls, 
#             input_size, 
#             joint_feature_size, 
#             output_size, 
#             flattened_mask_matrices: torch.Tensor) -> "ProdCollectiveAS":
#         connectivity_input_joint, connectivity_joint_output = cls.from_flattened_mask_matrices(
#             input_size=input_size,
#             joint_feature_size=joint_feature_size,
#             output_size=output_size,
#             flattened_mask_matrices=flattened_mask_matrices
#         )
#         return ProdCollectiveAS(
#             input_size=input_size, 
#             joint_feature_size=joint_feature_size, 
#             output_size=output_size, 
#             connectivity_input_joint=connectivity_input_joint, 
#             connectivity_joint_output=connectivity_joint_output)

#     def forward(self, x):
#         x = self.prod(x)
#         x = self.sparse_linear(x)
#         return x
    
#     def get_connectivities(self):
#         return [self.prod.get_connectivities(), self.sparse_linear.get_connectivities()]

#     def get_mask_matrix_encoding(self) -> List[torch.Tensor]:
#         return [self.prod.get_mask_matrix_encoding(), self.sparse_linear.get_mask_matrix_encoding()]

#     @classmethod
#     def from_mask_matrix_encoding_to_connectivities(self, mask_matrix_encoding, input_size, joint_feature_size, output_size):
#         prod_encoding, sparse_linear_encoding = mask_matrix_encoding
#         prod_connectivity = mask_matrix_2_connectivity_list(
#             prod_encoding.reshape(input_size, joint_feature_size)
#         )
#         sparse_linear_connectivity = mask_matrix_2_connectivity_list(
#             sparse_linear_encoding.reshape(joint_feature_size, output_size)
#         )
#         return (prod_connectivity, sparse_linear_connectivity)




class JASGBAG(SparseABC):
    """
    Implementation of a Gradual Bipolar Argumentation Graph / edge-weighted QBAF with joint support attack
    """
    def __init__(self, 
        no_softmax=False,
        *,
        input_size, 
        hidden_size, 
        output_size, 
        joint_connection_size1,
        joint_connection_size2,
        connections_input_hidden,
        connections_hidden_output,
        connections_jointly_input_hidden,
        connections_jointly_hidden_output,
    ):
        super().__init__()

        self.fitness = torch.tensor(0)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.joint_connection_size1 = joint_connection_size1
        self.joint_connection_size2 = joint_connection_size2

        self.sparse_linear1 = SparseLinearEnhanced(
            input_size, 
            hidden_size, 
            connectivity=connections_input_hidden)
        self.collective1 = ProdCollectiveAS(
            input_size=input_size,
            output_size=joint_connection_size1,
            connectivities=connections_jointly_input_hidden)
        self.activation1 = nn.Sigmoid()

        self.sparse_linear2 = SparseLinearEnhanced(
            hidden_size, 
            output_size, 
            connectivity=connections_hidden_output)
        self.collective2 = ProdCollectiveAS(
            input_size=hidden_size,
            output_size=joint_connection_size2,
            connectivities=connections_jointly_hidden_output)
        if no_softmax:
            self.output_layer = lambda x: x
        else:
            self.output_layer = nn.Softmax()
    
    def reduced_num_conn(self):
        connectivity2_all = torch.hstack((
            self.sparse_linear2.connectivity,
            self.collective2.get_connectivities()))
        sparse_linear1_before_remove = self.sparse_linear1.get_connectivities().clone()
        sparse_linear2_before_remove = self.sparse_linear2.get_connectivities().clone()
        collective1_before_remove = self.collective1.get_connectivities().clone()
        collective2_before_remove = self.collective2.get_connectivities().clone()

        connected_hidden_neurons_1 = set()

        sparse_linear1_after_remove = set()
        sparse_linear2_after_remove = set() 
        collective1_after_remove = set()
        collective2_after_remove = set()

        for conn in sparse_linear1_before_remove.T:
            conn = tuple(conn.tolist())
            to_idx, from_idx = conn
            if to_idx in connectivity2_all[1]:
                sparse_linear1_after_remove.add(conn)

        for conn in collective1_before_remove.T:
            conn = tuple(conn.tolist())
            to_idx, from_idx = conn
            if to_idx in connectivity2_all[1]:
                collective1_after_remove.add(conn)

        for conn in sparse_linear1_after_remove:
            to_idx, from_idx = conn
            connected_hidden_neurons_1.add(to_idx)

        for conn in collective1_after_remove:
            to_idx, from_idx = conn
            connected_hidden_neurons_1.add(to_idx)

        for conn in sparse_linear2_before_remove.T:
            conn = tuple(conn.tolist())
            to_idx, from_idx = conn
            if from_idx in connected_hidden_neurons_1:
                sparse_linear2_after_remove.add(conn)

        for conn in collective2_before_remove.T:
            conn = tuple(conn.tolist())
            to_idx, from_idx = conn
            if from_idx in connected_hidden_neurons_1:
                collective2_after_remove.add(conn)

        n_sparse_conns = len(sparse_linear1_after_remove) + len(sparse_linear2_after_remove)
        collective_conns1 = { to_idx for (to_idx, _) in collective1_after_remove }
        collective_conns2 = { to_idx for (to_idx, _) in collective2_after_remove }
        n_collective_conns = len(collective_conns1) + len(collective_conns2)
        return n_sparse_conns + n_collective_conns

    def describe(self):
        return {
            "sparse_linear1": self.sparse_linear1.describe(),
            "collective1": self.collective1.describe(),
            "sparse_linear2": self.sparse_linear2.describe(),
            "collective2": self.collective2.describe(),
        }

    def total_max_num_conn(self):
        return self.sparse_linear1.total_max_num_conn() + self.collective1.total_max_num_conn() + self.sparse_linear2.total_max_num_conn() + self.collective2.total_max_num_conn()
    
    def total_num_conn(self):
        return self.sparse_linear1.total_num_conn() + self.collective1.total_num_conn() + self.sparse_linear2.total_num_conn() + self.collective2.total_num_conn()

    @classmethod
    def random_connectivity_init(cls, params, no_softmax=False):
        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        output_size = params["output_size"]
        joint_connection_size1 = params["joint_connections_size1"]
        joint_connection_size2 = params["joint_connections_size2"]
        n_connections_input_hidden = params["number_connections1"]
        n_connections_hidden_output = params["number_connections2"]
        n_connections_jointly_input_hidden = params["joint_connections_input_num1"]
        n_connections_jointly_hidden_output = params["joint_connections_input_num2"]

        connectivity_input_hidden = connectivity_list_random(input_size, hidden_size, n_connections_input_hidden)
        connectivity_hidden_output = connectivity_list_random(hidden_size, output_size, n_connections_hidden_output)
        connectivity_jointly_input_hidden = connectivity_list_random(input_size, joint_connection_size1, n_connections_jointly_input_hidden)
        connectivity_jointly_hidden_output = connectivity_list_random(hidden_size, joint_connection_size2, n_connections_jointly_hidden_output)
        return JASGBAG(
            no_softmax=no_softmax,
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            joint_connection_size1=joint_connection_size1,
            joint_connection_size2=joint_connection_size2,
            connections_input_hidden=connectivity_input_hidden,
            connections_hidden_output=connectivity_hidden_output,
            connections_jointly_input_hidden=connectivity_jointly_input_hidden,
            connections_jointly_hidden_output=connectivity_jointly_hidden_output,
        )

    @classmethod
    def from_mask_matrix_encoding_to_connectivities(
            self,
            *,
            input_size:int,
            hidden_size:int,
            output_size:int,
            joint_connection_size1,
            joint_connection_size2,
            mask_matrix_encoding) -> List[torch.Tensor]:
        encoding_sparse_linear1 = mask_matrix_encoding[0]
        encoding_collective1 = mask_matrix_encoding[1]
        encoding_sparse_linear2 = mask_matrix_encoding[2]
        encoding_collective2 = mask_matrix_encoding[3]

        connectivity_sparse_linear1 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            mask_matrix_encoding=encoding_sparse_linear1,
            input_size=input_size,
            output_size=hidden_size
        )
        connectivities_collective1 = ProdCollectiveAS.from_mask_matrix_encoding_to_connectivity(
            mask_matrix_encoding=encoding_collective1,
            input_size=input_size,
            output_size=joint_connection_size1
        )
        connectivity_sparse_linear2 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            mask_matrix_encoding=encoding_sparse_linear2,
            input_size=hidden_size,
            output_size=output_size
        )
        connectivities_collective2 = ProdCollectiveAS.from_mask_matrix_encoding_to_connectivity(
            mask_matrix_encoding=encoding_collective2,
            input_size=hidden_size,
            output_size=joint_connection_size2
        )
        return (connectivity_sparse_linear1, connectivities_collective1, connectivity_sparse_linear2, connectivities_collective2)

    @classmethod
    def from_mask_matrix_encoding(
            cls, 
            params,
            mask_matrix_encoding : List[torch.Tensor], 
            ):
        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        output_size = params["output_size"]
        joint_connection_size1 = params["joint_connections_size1"]
        joint_connection_size2 = params["joint_connections_size2"]
        (connectivity_sparse_linear1, connectivity_jointly_input_hidden , 
         connectivity_sparse_linear2, connectivity_jointly_hidden_output)  = JASGBAG.from_mask_matrix_encoding_to_connectivities(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            joint_connection_size1=joint_connection_size1,
            joint_connection_size2=joint_connection_size2,
            mask_matrix_encoding=mask_matrix_encoding
        )

        return JASGBAG(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            joint_connection_size1=joint_connection_size1,
            joint_connection_size2=joint_connection_size2,
            connections_input_hidden=connectivity_sparse_linear1,
            connections_hidden_output=connectivity_sparse_linear2,
            connections_jointly_input_hidden=connectivity_jointly_input_hidden,
            connections_jointly_hidden_output=connectivity_jointly_hidden_output)

    def get_connectivities(self):
        return (self.sparse_linear1.get_connectivities(),
            self.collective1.get_connectivities(),
            self.sparse_linear2.get_connectivities(),
            self.collective2.get_connectivities())

    def get_mask_matrix_encoding(self):
        return [self.sparse_linear1.get_mask_matrix_encoding(),
            self.collective1.get_mask_matrix_encoding(),
            self.sparse_linear2.get_mask_matrix_encoding(),
            self.collective2.get_mask_matrix_encoding()]

    def forward(self, x):
        x1 = self.sparse_linear1(x)
        x2 = self.collective1(x)
        x = x1
        x[:, :self.joint_connection_size1] += x2
        x = self.activation1(x)
        x1 = self.sparse_linear2(x)
        x2 = self.collective2(x)
        x = x1
        x[:, :self.joint_connection_size2] += x2
        x = self.output_layer(x)
        return x
    


def test_jasgbag():
    input_size = 5
    hidden_size = 10
    output_size = 2
    connectivity_input_hidden = torch.tensor([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ], dtype=torch.float32)
    connectivity_hidden_output = torch.tensor([
        [1, 1],
        [0, 0]
    ], dtype=torch.float32)
    connectivity_jointly_input_hidden = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1]
    ], dtype=torch.float32)
    connectivity_jointly_hidden_output = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ], dtype=torch.float32)
    jasgbag = JASGBAG(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        connections_input_hidden=connectivity_input_hidden,
        connections_hidden_output=connectivity_hidden_output,
        connections_jointly_input_hidden=connectivity_jointly_input_hidden,
        connections_jointly_hidden_output=connectivity_jointly_hidden_output)

    connectivities = [
        connectivity_input_hidden,
        connectivity_jointly_input_hidden,
        connectivity_hidden_output,
        connectivity_jointly_hidden_output
    ]

    x = torch.ones((5, 5))
    y = jasgbag(x)
    assert y.shape == (5, 2)
    # assert torch.allclose(y, torch.ones((5, 2)))
    for i, j in zip(jasgbag.get_connectivities(), connectivities):
        i = i.to(torch.float32)
        j = j.to(torch.float32)
        assert torch.allclose(i, j)

    matrix_encodings = jasgbag.get_mask_matrix_encoding()
    jasgbag2 = JASGBAG.from_flattened_mask_matrices(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        mask_matrix_encoding=matrix_encodings)
    a, b, c, d = jasgbag2.get_connectivities() 
    assert torch.allclose(a, torch.tensor([[1], [0]]))
    assert torch.allclose(b, torch.tensor([[0, 1, 1], [0, 0, 1]]))
    assert torch.allclose(c, torch.tensor([[1], [0]]))
    assert torch.allclose(d, torch.tensor([[1, 1], [0, 1]]))


def test_adjacency_matrix_2_list_conversion():
    adjacency_matrix = torch.tensor([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=torch.float32)
    adjacency_list = mask_matrix_2_connectivity_list(adjacency_matrix)
    expected_adjacency_list = torch.tensor([[1, 2, 0, 2, 0, 1, 3, 2, 4, 3],
        [0, 0, 1, 1, 2, 2, 2, 3, 3, 4]])
    assert torch.allclose(expected_adjacency_list, adjacency_list)

def test_adjacency_list_2_matrix_conversion():
    adjacency_list = torch.tensor([[1, 2, 0, 2, 0, 1, 3, 2, 4, 3],
        [0, 0, 1, 1, 2, 2, 2, 3, 3, 4]])
    adjacency_matrix = adjacency_list_2_mask_matrix(adjacency_list, 5, 5)
    expected_adjacency_matrix = torch.tensor([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=torch.int)
    assert torch.allclose(expected_adjacency_matrix, adjacency_matrix)

if __name__ == "__main__":
    test_adjacency_list_2_matrix_conversion()
    print("test adjacency list 2 matrix conversion passed")
    test_adjacency_matrix_2_list_conversion()
    print("test adjacency matrix 2 list conversion passed")
    test_prod_collective_as()
    print("test prod collective as passed")
    test_jasgbag()
    print("test jasgbag passed")
    # x = JASGBAG.random_connectivity_init(
    #     input_size=10,
    #     hidden_size=10,
    #     output_size=10,
    #     joint_feature_size1=10,
    #     joint_feature_size2=10,
    #     n_connections_input_hidden=10,
    #     n_connections_hidden_output=10,
    #     n_connections_input_joint1=10,
    #     n_connections_joint1_hidden=10,
    #     n_connections_hidden_joint2=10,
    #     n_connections_joint2_output=10,
    # )
    # z = x.get_mask_matrix_encoding()
    # print(x.get_mask_matrix_encoding())
    # print(z)
    # y = JASGBAG.from_flattened_mask_matrices(
    #     input_size=10,
    #     hidden_size=10,
    #     output_size=10,
    #     joint_feature_size1=10,
    #     joint_feature_size2=10,
    #     mask_matrix_encoding=z
    # )
    # print(y.get_mask_matrix_encoding())
    # def recur_all_eq(l, r):
    #     if isinstance(l, torch.Tensor):
    #         return torch.allclose(l, r)
    #     if isinstance(l, list):
    #         return all(recur_all_eq(x, y) for x, y in zip(l, r))
    # print(recur_all_eq(y.get_mask_matrix_encoding(), z))
    