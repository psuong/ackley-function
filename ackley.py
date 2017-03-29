import numpy as np

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """
    x: vector of input values
    """
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

# The values in the array should range between -32 and 32
MIN_RANGE = -32
MAX_RANGE = 32


def rand_sample(rand_seed, num_samples):
    d = len(rand_seed)
    min_xs = rand_seed
    min_y = ackley(rand_seed)

    for _ in range(num_samples):
        curr_x = np.random.rand(d) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE
        curr_y = ackley(curr_x)
        if curr_y < min_y:
            min_xs = curr_x
            min_y = curr_y

    return (min_xs, min_y)


def best_neighbor(min_xs, min_y, x_idx, step_size):
    next_xs = np.copy(min_xs)
    prev_xs = np.copy(min_xs)
    next_xs[x_idx] += step_size
    prev_xs[x_idx] -= step_size

    next_y = ackley(next_xs)
    prev_y = ackley(prev_xs)
    next_y_diff = min_y - next_y
    prev_y_diff = min_y - prev_y

    curr_diff = 0

    if prev_y_diff > next_y_diff:
        if prev_y_diff > 0:
            min_xs = prev_xs
            min_y = prev_y
            curr_diff = prev_y_diff
    elif prev_y_diff < next_y_diff:
        if next_y_diff > 0:
            min_xs = next_xs
            min_y = next_y
            curr_diff = next_y_diff
    else:
        if prev_y_diff > 0:
            r = np.random.rand()
            if r < 0.5:
                min_xs = prev_xs
                min_y = prev_y
                curr_diff = prev_y_diff
            else:
                min_xs = next_xs
                min_y = next_y
                curr_diff = next_y_diff
    return (min_xs, min_y, curr_diff)


def hill_climb(rand_seed, step_size, epsilon=1e-9):
    d = len(rand_seed)
    min_xs = rand_seed
    min_y = ackley(rand_seed)

    for x_idx in range(d):
        curr_diff = np.inf
        num_iter = 0
        while curr_diff > epsilon and num_iter < 10000:
            min_xs, min_y, curr_diff = best_neighbor(min_xs, min_y, x_idx, step_size)
            num_iter += 1

    return (min_xs, min_y)

def temperature(curr, total):
    return 1 - curr/total

def simulated_annealing(rand_seed, step_size, num_samples, epsilon=1e-9):
    d = len(rand_seed)
    min_xs = rand_seed
    min_y = ackley(rand_seed)

    for x_idx in range(d):
        curr_xs = min_xs
        curr_y = min_y

        for i in range(num_samples):
            T = temperature(i, num_samples)
            rand = np.random.random()
            if T > rand:
                curr_xs[x_idx] = np.random.random() * (MAX_RANGE - MIN_RANGE) + MIN_RANGE
                curr_y = ackley(curr_xs)
                if curr_y < min_y:
                    min_xs = curr_xs
                    min_y = curr_y
            else:
                min_xs, min_y, curr_diff = best_neighbor(min_xs, min_y, x_idx, step_size)

    return (min_xs, min_y)


class Particle:
    """
    Particle for PSO algorithm
    """
    def __init__(self, dim):
        self.dim = dim
        self.position = np.random.rand(dim) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE
        self.best_position = self.position
        self.best_value = ackley(self.position)
        self.velocity = np.random.rand(dim) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE

    def update_velocity(self, c, cp, cg, g):
        for d in range(self.dim):
            rp = np.random.random()
            rg = np.random.random()
            self.velocity[d] = (c * self.velocity[d] + 
                                cp*rp*(self.best_position[d] - self.position[d]) +
                                cg*rg*(g[d] - self.position[d]))
    def update_position(self):
        self.position += self.velocity
        new_value = ackley(self.position)
        if new_value < self.best_value:
            self.best_position = self.position
            self.best_value = new_value
        return (self.position, self.best_value)

def pso(num_iter, num_particles, dim, c, cp, cg):
    """
    Particle Swarm Optimization
    """
    seed_p = Particle(dim)
    particles = [seed_p]

    best_global_position = seed_p.best_position
    best_global_value = seed_p.best_value

    for _ in range(num_particles-1):
        p = Particle(dim)
        if p.best_value < best_global_value:
            best_global_position = p.best_position
            best_global_value = p.best_value

        particles.append(p)

    for _ in range(num_iter):
        for particle in particles:
            particle.update_velocity(c, cp, cg, best_global_position)
            new_position, new_value = particle.update_position()
            if new_value < best_global_value:
                best_global_position = new_position
                best_global_value = new_value

    return (best_global_position, best_global_value)


if __name__=="__main__":
    dim = 2
    rand_seed = np.random.rand(dim) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE
    print("                                      Random Seed:", (rand_seed, ackley(rand_seed)))
    print("                    Random Sampling (100 samples):", rand_sample(rand_seed, 100))
    print("                   Random Sampling (1000 samples):", rand_sample(rand_seed, 1000))
    print("                  Random Sampling (10000 samples):", rand_sample(rand_seed, 10000))
    print("                         Hill Climbing (0.1 step):", hill_climb(rand_seed, step_size=0.1))
    print("           Simulated Annealing (0.1 step, 10000s):", simulated_annealing(rand_seed, step_size=0.1, num_samples=10000))
    print("PSO (num_iter=100,part=1000, c,cp,cg=0.5,0.5,0.1):", pso(num_iter=100, num_particles=1000, dim=dim, c=0.5, cp=0.5, cg=0.1))


