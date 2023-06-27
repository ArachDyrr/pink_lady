from modules.myCNN import MyCNN_97531_max

model = MyCNN_97531_max(.1)

print (model)

# Count the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Calculate the memory footprint assuming float32 precision
memory_footprint = total_params * 4 / (1024 ** 2)  # Convert to megabytes
print(f"Memory footprint (float32 precision): {memory_footprint} MB")
print()
x = model.get_flattensize()
print(x)
print()

layer_1 = 9*9*(224*224)*3*64
layer_2 = 7*7*(224*224)*64*128
layer_3 = 5*5*(224*224)*128*256
layer_4 = 3*3*(224*224)*256*512
layer_5 = 1*1*(224*224)*512*512
layer_conv = 1*1*(224*224)*512*4



print (f'layer_1= {layer_1}')
print (f'layer_2= {layer_2}')
print (f'layer_3= {layer_3}')
print (f'layer_4= {layer_4}')
print (f'layer_5= {layer_5}')
print (f'layer_conv= {layer_conv}')

total = layer_1 + layer_2 + layer_3 + layer_4 + layer_5 + layer_conv
print (f'total= {total}')
