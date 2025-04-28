# my_agent_app.py
from agents import (
    Agent,
    LLMClient,
    FinalAnswer,
    GetUserInput,
    MakePlan,
    WriteFile,
    ReadFile,
    EditFile,
    RunPython,
    RunBash,
    ListFiles,
    Delete,
    ViewImage,
)



if __name__ == "__main__":
    import argparse, sys, os

    parser = argparse.ArgumentParser(description="Run multi-agent tool-calling assistant")
    parser.add_argument("task", nargs="?", help="Initial user task (if omitted you will be prompted)")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Verbose OpenAI request/response logging")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use a local LLM instead of the OpenAI API for the executor agent")
    parser.add_argument("-v", "--verbosity", type=int, default=3, choices=range(0,4),
                        help="Verbosity level: 0-quiet 1-steps 2-rich 3-trace (default 2)")
    parser.add_argument("-c", "--confirm-edits", action="store_true", 
                        help="Require confirmation before editing or deleting files")
    parser.add_argument("--confirm-plan", action="store_true", 
                        help="Ask for confirmation after making initial plan")
    parser.add_argument( "--end", action="store_true", 
                        help="End on first final_answer")

    cli_args = parser.parse_args()

    # Set confirm_edits variable based on command line argument
    confirm_edits = cli_args.confirm_edits

    # ------------------ (2a)  Build TWO model objects ----------------------

    # ––– Executor can use either the OpenAI mini model or a local server –––
    if cli_args.local:
        model_executor = LLMClient(
            model_id="lmstudio",
            debug=cli_args.debug,
            api_base="http://localhost:1234/v1",
        )
        model_planner = LLMClient(
            model_id="lmstudio",
            debug=cli_args.debug,
            api_base="http://localhost:1234/v1",
        )
    else:
        model_executor = LLMClient(
            model_id="gpt-4.1",
            debug=cli_args.debug,
        )
        # ––– Planner uses the stronger "o3" model (always OpenAI for now) –––
        model_planner = LLMClient(model_id="o3", temperature=1, debug=cli_args.debug)

    # ------------------ (2e)  Get the initial user task --------------------

    # If task ends with .txt, assume it's a file path and read from it
    if cli_args.task and cli_args.task.endswith(".txt"):
        user_task = open(cli_args.task).read()
    elif cli_args.task:
        user_task = cli_args.task
    else:
        # Fall back to prompting if no task provided
        user_task = input("Enter your task: ")


    import os
    from datetime import datetime
    cwd = None
    if cwd is None:
        cwd = os.getcwd()

    workdir = os.path.join(cwd,'work/tmp_'+datetime.now().strftime('%y%m%d_%H%M'))
    os.makedirs(workdir,exist_ok=True)
    print(f'moving to {workdir}')
    os.chdir(workdir)



    # ------------------ (2b)  Shared tool belt -----------------------------

    tools = [
        WriteFile(), ReadFile(), EditFile(confirm_edits=confirm_edits), RunPython(),
        RunBash(), Delete(confirm_edits=confirm_edits), ViewImage(),
        MakePlan(), ListFiles(), FinalAnswer()
    ]

    tools_manager = [
         MakePlan(), FinalAnswer(),
    ]
    if cli_args.confirm_plan:
        tools_manager.append(GetUserInput())

    
    tools_mentor = [
        ListFiles(), ReadFile(), FinalAnswer(),
    ]

    # ------------------ (2c) Build MENTOR agent ---------------------------

    mentor_prompt = ("You are the *mentor_agent* - an expert software engineer with comprehensive knowledge "
                     "and experience. Your role is to provide guidance, advice, and help when the executor_agent "
                     "is struggling with a problem. You have access to the codebase through the ReadFile tool, "
                     "so you can examine relevant files and provide informed advice based on the current state "
                     "of the project. You start with a clean memory for each task, so you need comprehensive context "
                     "about the current problem to provide effective guidance.")
                     
    mentor_agent = Agent(
        tools=tools_mentor,
        model=model_planner,  # Using the stronger model for mentoring
        system_message=mentor_prompt,
        verbosity=cli_args.verbosity,
        max_steps=20,
        name="mentor_agent",
        description="Provides expert guidance and advice when executor_agent is struggling with a problem.",
        clear_memory_on_run=True,  # Starts with a clean memory for each consultation
    )

    # ------------------ (2d)  Build EXECUTOR agent -------------------------

    executor_prompt = ("You are the *executor_agent* - a skilled implementer. Your job is to complete "
                     "individual tasks assigned by the manager agent using the provided tools. If you find "
                     "yourself struggling with the same problem a couple times, "
                     "consult the mentor_agent by providing a clear description of the problem "
                     "you're facing, including any error messages and relevant code snippets.")

    executor_agent = Agent(
        tools=tools,
        model=model_executor,
        system_message=executor_prompt,
        planning_interval=None,          # executor just acts, no extra planning
        verbosity=cli_args.verbosity,
        max_steps=30,
        managed_agents=[mentor_agent],   # Give executor access to the mentor
        name="executor_agent",
        description="Executes a single step using the provided tool belt and returns the observation.",
        clear_memory_on_run=True,  # Starts with a clean memory for each task

    )

    # ------------------ (2d)  Build MANAGER (planner) agent ----------------

    manager_prompt = ("You are the *manager_agent* - a senior engineer. Your first "
            "task is to use the make_plan tool. In the plan you should start with "
            "Goal: <the overall goal of the task -- restating it in your own words> "
            "Completion Criteria: <what did the user specify that woiuld complete the task> "
            "Next, break the user's high-level request into a numbered sequence of "
            "concrete, executable steps. Each step MUST include: "
            "(1) a clear instruction for the executor agent "
            "(2) any parameters or file names needed, and "
            "(3) explicit completion criteria. You are the expert in this area, "
            "so be very clear about the details and instructions so the executor doesn't "
            "have to fill in too many blanks. "
            "Your plan should not be overly complex -- accomplish the task in the minimal "
            "number of steps necessary. Do NOT ask the agent to write scaffolding files first. " 
            "Do NOT ask the agent not use a virtual environment or git repo or install anyything."
            ""
            "Then you must delegate the *first* step to `executor_agent` and "
            "wait for its report. When a step is reported complete, mark it as "
            "done ✅ and delegate the next. Repeat until all steps are finished. "
            "For the Second step and beyond, the executor_agent does not know of any "
            "previous work so be sure to give complete and verbose context. "
            "NEVER call final_answer until all steps are reported ✅ complete by "
            "executor_agent. If you think work is finished, first verify each "
            "criterion, then call final_answer.")

    if cli_args.confirm_plan:
            manager_prompt+="\n\nAfter creating the plan, ask the user for any further changes or approval."


    manager_agent = Agent(
        tools=tools_manager,
        model=model_planner,
        system_message=manager_prompt,
        managed_agents=[executor_agent],
        planning_interval=None,         # always plan/update each cycle
        verbosity=cli_args.verbosity,
        max_steps=50,
        name="manager_agent",
        description="Decomposes the task and coordinates with executor_agent.",
    )

    # ------------------ (2f)  Run the MANAGER agent ------------------------


    final_response = manager_agent.run(user_task)
    print("\n=== FINAL ANSWER ===\n" + final_response)

    # Optional interactive feedback loop remains available (now routed to the
    # manager so it can decide if more work is needed).
    while True and not cli_args.end:
        follow_up = input("Feedback (or 'end'): ")
        if follow_up.strip().lower() == "end":
            break
        # give the feedback to the *manager*; let it decide what to do
        manager_agent.max_steps += 20
        print(manager_agent.run(follow_up, reset=False))

    os.chdir(cwd)

