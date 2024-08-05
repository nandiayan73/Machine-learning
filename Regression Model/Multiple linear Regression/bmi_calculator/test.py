# weight (kg) / [height (m)]2
weights=[69,55,66,43,89,102,55,66,88,99]
height=[1.67,1.77,1.89,1.56,1.89,1.78,1.91,1.66,1.56,1.98]
i=0
for weight in weights:
    bmi=weight/(height[i]*height[i])
    print(weight,end=",")
    print(height[i],end=",")
    print(bmi)
    i+=1