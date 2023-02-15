from collections import Counter
import matplotlib.pyplot as plt

def drawGraph(dict, result, dataarray):

    # https://www.geeksforgeeks.org/python-program-to-find-the-highest-3-values-in-a-dictionary/
    # Found most common 10 words and their number of uses
    dict_ten = Counter(dict)
    dict_ten = dict_ten.most_common(10)


    # https://stackoverflow.com/questions/62593913/plotting-a-pie-chart-out-of-a-dictionary
    labels = []
    sizes = []

    for x, y in dict_ten:
        labels.append(x)
        sizes.append(y)

    # Plot and explode most common 3 words
    plt.pie(sizes, explode=[0.2 if n < 3 else 0 for n in range(10)], labels=labels, autopct='%1.2f%%', shadow=True)
    plt.title(f"Most Common 10 words in {result}")

    plt.axis('equal')
    plt.show()
    print(f"For {result} Text Most Common 3 Words:")

    for i, j in dict_ten[:3]:
        statistic = 0
        count = 0
        for k in dataarray:
            if k[1] == result:
                count+=1
                if i in k[0]:
                    statistic += 1
        print("Word:", i, "Number of word:", j, "Statistic is:", (statistic/count))
