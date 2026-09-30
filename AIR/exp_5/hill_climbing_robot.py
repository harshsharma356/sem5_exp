import math
import matplotlib.pyplot as plt

grid_size = 8

start = (1, 1)
goal = (8, 8)

obstacles = {
    (3,5), (4,5), (5,5),
    (5,6), (6,6), (7,6),
    (2,3), (2,4),
    (3,3), (3,4)
}



def cost(position):
    x1, y1 = position
    x2, y2 = goal

    return math.sqrt(
        (x2-x1)**2 +
        (y2-y1)**2
    )


moves = [
    (-1,0),   
    (1,0),    
    (0,-1),   
    (0,1),    
    (-1,-1),  
    (-1,1),
    (1,-1),
    (1,1)
]



def get_neighbors(position):

    neighbours = []

    x,y = position

    for dx,dy in moves:

        nx = x + dx
        ny = y + dy

        if nx < 1 or nx > grid_size:
            continue

        if ny < 1 or ny > grid_size:
            continue


        new_position = (nx,ny)

        if new_position in obstacles:
            continue


        neighbours.append(new_position)


    return neighbours

def hill_climbing():

    current = start

    path = [current]

    observations = []


    step = 1


    while current != goal:


        neighbours = get_neighbors(current)


        if len(neighbours)==0:
            print("No valid moves")
            break


        next_position = min(
            neighbours,
            key=cost
        )


        current_cost = cost(current)

        next_cost = cost(next_position)


        if next_cost >= current_cost:

            print("Local minimum reached")

            break


        observations.append(
            [
                step,
                current,
                next_position,
                round(next_cost,3)
            ]
        )


        current = next_position

        path.append(current)


        step += 1



    return path, observations


path, observations = hill_climbing()

print("\nRobot Path")

for p in path:
    print(p)


print("\nIteration Details")

print(
    "Step\tCurrent\t\tSelected\tCost"
)


for row in observations:

    print(
        row[0],
        "\t",
        row[1],
        "\t",
        row[2],
        "\t",
        row[3]
    )


print("\nFinal Position:", path[-1])

print(
    "Goal Reached:",
    path[-1]==goal
)


print(
    "Number of Steps:",
    len(path)-1
)


plt.figure(figsize=(7,7))


for i in range(1,9):

    plt.axhline(i-0.5,color='gray')
    plt.axvline(i-0.5,color='gray')


for obs in obstacles:

    plt.scatter(
        obs[0],
        obs[1],
        marker="s",
        s=500
    )


px = [p[0] for p in path]
py = [p[1] for p in path]


plt.plot(
    px,
    py,
    marker="o",
    linewidth=2
)



plt.scatter(
    start[0],
    start[1],
    s=300,
    marker="o"
)



plt.scatter(
    goal[0],
    goal[1],
    s=300,
    marker="*"
)

plt.xlim(0.5,8.5)
plt.ylim(0.5,8.5)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "Robot Path using Hill Climbing Algorithm"
)

plt.grid()

plt.show()