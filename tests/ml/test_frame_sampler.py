from ml.data.preprocessing.frame_sampler import sample_frame_indices

indices = sample_frame_indices(300, 16)

print("Indices:", indices)
print("Number of samples:", len(indices))
print("First frame:", indices[0])
print("Last frame:", indices[-1])
