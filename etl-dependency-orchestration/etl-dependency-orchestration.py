def schedule_pipeline(tasks: list, resource_budget: int) -> list:
    """
    Returns a list of schedule dictionaries.
    """
    task_map = {
        task["name"]: task
        for task in tasks
    }

    completed = set()
    started = set()

    # task_name -> end_time
    running = {}

    schedule = []
    current_time = 0

    while len(completed) < len(tasks):

        # -------------------------------------------------
        # 1. Complete every running task whose end time
        #    has been reached.
        # -------------------------------------------------
        for name, end_time in list(running.items()):
            if end_time <= current_time:
                completed.add(name)
                del running[name]

        # -------------------------------------------------
        # 2. Find ready tasks.
        # -------------------------------------------------
        ready = [
            name
            for name, task in task_map.items()
            if (
                name not in started
                and all(
                    dependency in completed
                    for dependency in task["depends_on"]
                )
            )
        ]

        # -------------------------------------------------
        # 3. Alphabetical order.
        # -------------------------------------------------
        ready.sort()

        # Current resource usage.
        used_resources = sum(
            task_map[name]["resources"]
            for name in running
        )

        # -------------------------------------------------
        # 4. Greedily start tasks that fit.
        # -------------------------------------------------
        for name in ready:
            required = task_map[name]["resources"]

            if used_resources + required <= resource_budget:
                running[name] = (
                    current_time
                    + task_map[name]["duration"]
                )

                started.add(name)
                used_resources += required

                schedule.append({
                    "task_name": name,
                    "start_time": current_time
                })

        # -------------------------------------------------
        # 5. Stop if everything is completed.
        # -------------------------------------------------
        if len(completed) == len(tasks):
            break

        # -------------------------------------------------
        # 6. Advance to the next completion event.
        # -------------------------------------------------
        if running:
            current_time = min(running.values())
        else:
            # Valid test cases should not reach this state.
            break

    # Required final ordering.
    return sorted(
        schedule,
        key=lambda item: (
            item["start_time"],
            item["task_name"]
        )
    )