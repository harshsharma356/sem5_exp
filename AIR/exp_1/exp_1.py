import heapq

graph = {

    'A': {'B':2, 'C':3},

    'B': {'A':2, 'D':3, 'E':1},

    'C': {'A':3, 'F':2},

    'D': {'B':3, 'G':4},

    'E': {'B':1, 'G':3},

    'F': {'C':2, 'G':2},

    'G': {'D':4, 'E':3, 'F':2, 'H':1},

    'H': {'G':1}

}


heuristic = {

    'A':7,

    'B':6,

    'C':5,

    'D':4,

    'E':3,

    'F':2,

    'G':1,

    'H':0

}


def a_star(start, goal):

    open_list = []

    
    heapq.heappush(
        open_list,
        (heuristic[start], start)
    )


    came_from = {}

    g_cost = {

        start:0

    }


    while open_list:


        current = heapq.heappop(open_list)[1]


        if current == goal:

            path=[]

            while current in came_from:

                path.append(current)

                current=came_from[current]


            path.append(start)

            return path[::-1], g_cost[goal]



        for neighbor, cost in graph[current].items():


            new_g = g_cost[current] + cost


            if neighbor not in g_cost or new_g < g_cost[neighbor]:


                g_cost[neighbor] = new_g


                f = new_g + heuristic[neighbor]


                heapq.heappush(
                    open_list,
                    (f, neighbor)
                )


                came_from[neighbor]=current



    return None,None




path, cost = a_star('A','H')


print("Optimal Path:")

print(" -> ".join(path))


print("\nTotal Cost:")

print(cost)