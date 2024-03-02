import numpy as np
import asyncio
import aioconsole

from environment import Environment

async def read_input(timeout=1):
    try:
        input_data = await asyncio.wait_for(aioconsole.ainput(''), timeout=timeout)
        return input_data
    except asyncio.TimeoutError:
        return None

async def main():
    print("Press 'q' to quit the plot.")
    while True:
        action = 0 #2*(np.random.random()-0.5)
        world.step(action)
        world.visualize(action)

        # Read input asynchronously with a timeout
        input_data = await read_input(0.03)
        # Check if the input is 'esc', if so, break the loop
        if input_data is not None and input_data.lower() == 'q':
            print("Exiting the loop...")
            break

world = Environment(theta_init=np.pi/2)
world.set_reference([np.pi/3, 0])
asyncio.run(main())

