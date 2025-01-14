import matplotlib.pyplot as plt 
# Investment data 
investments = {'AMZ': 15.38, 'BTC': 7.69, 'Bonds': 76.92} 
# # Pie chart, where the slices will be ordered and plotted counter-clockwise: 
labels = investments.keys() 
sizes = investments.values() 
colors = ['#ff9999','#66b3ff','#99ff99'] 
fig1, ax1 = plt.subplots() 
ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90) 
# Equal aspect ratio ensures that pie chart is drawn as a circle. 
ax1.axis('equal') 
plt.title('Investment Distribution') 
plt.show()