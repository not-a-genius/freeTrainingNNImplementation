from __future__ import print_function
import torch
print("Is cuda available on this machine? ",
"Yes" if torch.cuda.is_available() else "No")

x = torch.rand(5, 3).cuda()
print("Using torch: ",x)


