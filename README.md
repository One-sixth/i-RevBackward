# i-RevBackward

This is a model similar to irevnet and follows the same principle as irevnet.  

In addition, I wrote the backward function so that it can really reduce the VRAM usage and can be connected with other irreversible modules.    

i-revnet is a very surprising method.  
This method saves a lot of video memory and allows me to train larger models.  

# Dependent
Currently I have tested it in pytorch_v1.3.1 pytorch_v1.5.1 pytorch_v1.6.  

# How it works
i-rev reversible method.  

```
On forward.
input x1, x2
y = x1 + F(x2)
output y, x2

On invert.
input y，x2
x1 = y - F(x2)
output x1, x2
```

# How to test
I used the cifar10 dataset for testing.  

1. Download this repository 
2. Run `python3 train_on_cifar10_with_rev_backward.py`  
3. Use nvidia-smi to observe how much video memory is used for training.  
4. Kill the program.  
5. Run `python3 train_on_cifar10_without_rev_backward.py`  
6. Check the VRAM usage again.  
7. Not surprisingly, the second VRAM occupies about twice as much as the first.  


# How to apply it to your own projects
The explanation is a bit difficult, I suggest you look directly at the code.  


# References
https://openreview.net/forum?id=HJsjkMb0Z  
https://github.com/jhjacobsen/pytorch-i-revnet  
