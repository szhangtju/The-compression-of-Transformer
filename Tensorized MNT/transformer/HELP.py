import torch 
Q = torch.randn(5, 9)
K = torch.randn(6, 5)
V = torch.randn(6, 5)

print(2/3)


# def self_attention(q,k,v):
# 	attn = torch.mm(q, k.transpose(0,1))
# 	output = torch.mm(attn, v)
# 	return output 
# def core_attention(q,k,v):
# 	core_value = torch.rand(5)
# 	core_tensor = torch.zeros(5,5,5)
# 	for i in range(5):
# 		for j in range(5):
# 			for k in range(5):
# 				if i==j==k:
# 					core_tensor[i,j,k] = core_value[i]
# 	A = torch.randn(3,5,4)
# 	l = torch.randn(2,5)
# 	r = torch.randn(2,4)
# 	return torch.einsum('bn,anm,bn->bab',l,A,r)

# 	# result = torch.einsum('bn,anm,bm->ba', k,q,v)
# 	# return result

# # print(self_attention(Q,K,V).size()) # (9,5)
# print(core_attention(Q,K,V).size())