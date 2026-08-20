import asyncio


async def main():
    print("NeuroFlow worker started.", flush=True)

    while True:
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())