# %%
import pandas as pd
from io import StringIO
df = pd.read_csv(StringIO(""",L,num_iterations,function_type,time_avg,time_mem_trsf_avg
0,10,200,numpyAdd,0.001283884048461914,0.0009417533874511719
1,100,200,numpyAdd,0.0006091594696044922,0.0005352497100830078
2,1000,200,numpyAdd,0.0010251998901367188,0.0008308887481689453
3,10000,200,numpyAdd,0.0026357173919677734,0.002491474151611328
4,100000,200,numpyAdd,0.042994022369384766,0.03977537155151367
5,1000000,200,numpyAdd,0.6350362300872803,0.6815159320831299
6,10000000,200,numpyAdd,15.003222227096558,15.305379629135132
7,100000000,200,numpyAdd,146.24223113059998,146.0736906528473
8,10,200,explicitAdd,0.030220000054687262,0.5258966401219368
9,100,200,explicitAdd,0.03004672009497885,0.5420569607615474
10,1000,200,explicitAdd,0.0299756800569594,0.5365747192502018
11,10000,200,explicitAdd,0.02948176013305787,0.5548615986108784
12,100000,200,explicitAdd,0.030781919984146956,0.7275027206540109
13,1000000,200,explicitAdd,0.04058080008253457,2.342787362337113
14,10000000,200,explicitAdd,0.05169856006279584,17.630777711868287
15,100000000,200,explicitAdd,0.053796159997582436,196.62277236938473
16,10,200,implicitAdd,0.4176623992621902,0.9095713603496549
17,100,200,implicitAdd,0.36061583995819085,0.8496516823768617
18,1000,200,implicitAdd,0.3598051196336747,0.8354521617293362
19,10000,200,implicitAdd,0.3832544001936914,0.8544886422157292
20,100000,200,implicitAdd,0.6901988780498504,1.1795977604389192
21,1000000,200,implicitAdd,3.6347342455387097,4.318892167806623
22,10000000,200,implicitAdd,36.58023847579959,37.97601020812989
23,100000000,200,implicitAdd,322.9234561157229,354.40112960815446
24,10,200,gpuarrayAdd_np,0.08144111938774586,0.3640982398390771
25,100,200,gpuarrayAdd_np,0.08405695945024487,0.3308627200126648
26,1000,200,gpuarrayAdd_np,0.08217119973152874,0.32901807948946954
27,10000,200,gpuarrayAdd_np,0.08439919959753744,0.35123967975378045
28,100000,200,gpuarrayAdd_np,0.06475216019898651,0.5121577574312689
29,1000000,200,gpuarrayAdd_np,0.2702651193737983,2.3907046365737914
30,10000000,200,gpuarrayAdd_np,0.9438420796394351,19.780379266738887
31,100000000,200,gpuarrayAdd_np,6.479453599452976,209.12848030090322
32,10,200,gpuarrayAdd,0.04423839997500183,0.826852159798145
33,100,200,gpuarrayAdd,0.045179519820958386,0.7681572777032859
34,1000,200,gpuarrayAdd,0.044544000010937425,0.7647087991237644
35,10000,200,gpuarrayAdd,0.04488320000469682,0.7889484795928001
36,100000,200,gpuarrayAdd,0.0462387199513614,0.9521072009205814
37,1000000,200,gpuarrayAdd,0.055970079898834256,2.8349518418312067
38,10000000,200,gpuarrayAdd,0.07454384010285137,19.951330089569097
39,100000000,200,gpuarrayAdd,0.07868719987571242,204.4933682250977"""), sep=",", )
df.to_csv("/Users/loreliegordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Fall2021/EECS4750/Assignments/assignment1/execution_time.csv")
# %%

import chartify

ch = chartify.Chart(blank_labels=True)
ch.set_title("Line charts - Grouped by color")
ch.plot.line(
    # Data must be sorted by x column
    data_frame=df,
    x_column='L',
    y_column='time_gpu_avg',
    color_column='function_type')
ch.show('png')

# %%