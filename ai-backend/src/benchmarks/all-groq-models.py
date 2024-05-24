from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

import time
from prettytable import PrettyTable

load_dotenv()


def batch():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    print(chain.invoke({"text": "Explain the importance of low latency LLMs."}))


# Streaming
def streaming():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a long poem about {topic}")]
    )
    chain = prompt | chat
    # for chunk in chain.stream({"topic": "The Moon"}):
    #     print(chunk.content, end="", flush=True)
    for chunk in chain.stream({"topic": "The Moon"}):
        print(chunk.content, end="", flush=True)


# def benchmark_response_time():
#     chat = ChatGroq(
#         temperature=0,
#         model_name="llama3-8b-8192",
#         groq_api_key=os.getenv("GROQ_API_KEY"),
#     )

#     system = "You are a helpful assistant."
#     human = "{text}"
#     prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

#     chain = prompt | chat

#     num_runs = 10
#     total_time = 0
#     for i in range(num_runs):
#         start_time = time.time()
#         response = chain.invoke({"text": "Explain the importance of low latency LLMs."})
#         end_time = time.time()

#         response_time = end_time - start_time
#         total_time += response_time

#         # num_tokens = len(response.split())
#         # tokens_per_second = num_tokens / response_time

#         print(f"Run {i+1}:")
#         print(f"Response Time: {response_time} seconds")
#         # print(f"Tokens per Second: {tokens_per_second}")

#     average_time = total_time / num_runs
#     print(f"\nAverage Response Time: {average_time} seconds")


def benchmark_response_time_streaming():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a long poem about {topic}")]
    )
    chain = prompt | chat

    num_runs = 10
    total_time = 0
    for i in range(num_runs):
        start_time = time.time()
        for chunk in chain.stream({"topic": "The Moon"}):
            pass
        end_time = time.time()

        response_time = end_time - start_time
        total_time += response_time

        print(f"Run {i+1}:")
        print(f"Response Time: {response_time} seconds")

    average_time = total_time / num_runs
    print(f"\nAverage Response Time: {average_time} seconds")


def benchmark_ttfb_streaming():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a long poem about {topic}")]
    )
    chain = prompt | chat

    num_runs = 10
    total_time = 0
    for i in range(num_runs):
        start_time = time.time()
        for chunk in chain.stream({"topic": "The Moon"}):
            end_time = time.time()
            break

        response_time = end_time - start_time
        total_time += response_time

        print(f"Run {i+1}:")
        print(f"Response Time: {response_time} seconds")

    average_time = total_time / num_runs
    print(f"\nAverage Response Time: {average_time} seconds")


def benchmark_ttfb_streaming_prettytable():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a long poem about {topic}")]
    )
    chain = prompt | chat

    num_runs = 5
    total_time = 0
    # Create a table with a column for each run
    table = PrettyTable()
    table.field_names = ["Run No", "Response Times (in ms)"]

    for i in range(num_runs):
        start_time = time.time()
        for chunk in chain.stream({"topic": "The Moon"}):
            end_time = time.time()
            break

        response_time = (end_time - start_time) * 1000
        total_time += response_time

        # Add the response time to the table
        table.add_row([f"Run {i+1}", f"{response_time:.3f}"])

    average_time = total_time / num_runs

    # Print the table
    print(table)
    print(f"\nAverage Response Time: {average_time:.3f} ms")


def llama3_streaming():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a short story about a journey to the {topic}")]
    )
    chain = prompt | chat

    num_runs = 2
    total_ttfb = 0
    total_response_time = 0

    # Create a table with three columns: "Run No", "TTFB", and "Total"
    table = PrettyTable()
    table.field_names = ["Run No", "TTFB (ms)", "TTLB (ms)"]

    for i in range(num_runs):
        # Measure TTFB and total response time
        start_time = time.time()
        first_byte_received = False
        for chunk in chain.stream({"topic": "center of the Earth"}):
            if not first_byte_received:
                end_time_ttfb = time.time()
                ttfb = (end_time_ttfb - start_time) * 1000  # Convert to milliseconds
                total_ttfb += ttfb
                first_byte_received = True
        end_time_total = time.time()
        total_time = (end_time_total - start_time) * 1000  # Convert to milliseconds
        total_response_time += total_time

        # Add the run number, TTFB, and total response time to the table
        table.add_row([i + 1, f"{ttfb:.3f}", f"{total_time:.3f}"])

    average_ttfb = total_ttfb / num_runs
    average_response_time = total_response_time / num_runs

    print("\nBenchmark Results:")
    print("=========================================")
    # Create a table for the model information
    model_table = PrettyTable()
    model_table.field_names = ["ID", "Name", "Provider", "Mode"]
    model_table.add_row(["llama3-8b-8192", "LLaMA 3 8B", "Groq", "Streaming"])
    # Print the model information table
    print("\nTable 1: Model Information:")
    print(model_table)

    # Print the first table
    print("\nTable 2: Response Times")
    print(table)

    # Create a second table with four columns: "No of runs", "Total Time Taken", "Average TTFB", "Average TTLB"
    table2 = PrettyTable()
    table2.field_names = [
        "No of runs",
        "Total Time Taken",
        "Average TTFB",
        "Average TTLB",
    ]
    table2.add_row(
        [
            num_runs,
            f"{total_response_time / 1000:.3f} seconds",
            f"{average_ttfb:.3f} ms",
            f"{average_response_time:.3f} ms",
        ]
    )

    # Print the second table
    print("\nTable 3: Benchmark Summary")
    print(table2)


def llama3_batch():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat

    num_runs = 2
    total_response_time = 0

    # Create a table with two columns: "Run No" and "TTLB"
    table = PrettyTable()
    table.field_names = ["Run No", "TTLB (ms)"]

    for i in range(num_runs):
        # Measure total response time
        start_time = time.time()
        chain.invoke(
            {"text": "Write a short story about a journey to the center of the Earth."}
        )
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_response_time += total_time

        # Add the run number and total response time to the table
        table.add_row([i + 1, f"{total_time:.3f}"])

    average_response_time = total_response_time / num_runs

    print("\nBenchmark Results:")
    print("=========================================")
    # Create a table for the model information
    model_table = PrettyTable()
    model_table.field_names = ["ID", "Name", "Provider", "Mode"]
    model_table.add_row(["llama3-8b-8192", "LLaMA 3 8B", "Groq", "Batch"])
    # Print the model information table
    print("\nTable 1: Model Information")
    print(model_table)

    # Print the first table
    print("\nTable 2: Response Times")
    print(table)

    # Create a second table with four columns: "No of runs", "Total Time Taken", "Average TTFB", "Average TTLB"
    table2 = PrettyTable()
    table2.field_names = [
        "No of runs",
        "Total Time Taken",
        "Average TTLB",
    ]
    table2.add_row(
        [
            num_runs,
            f"{total_response_time / 1000:.3f} seconds",
            f"{average_response_time:.3f} ms",
        ]
    )

    # Print the second table
    print("\nTable 3: Benchmark Summary")
    print(table2)


def llama3_70b_streaming():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a short story about a journey to the {topic}")]
    )
    chain = prompt | chat

    num_runs = 5
    total_ttfb = 0
    total_response_time = 0

    # Create a table with three columns: "Run No", "TTFB", and "Total"
    table = PrettyTable()
    table.field_names = ["Run No", "TTFB (ms)", "TTLB (ms)"]

    for i in range(num_runs):
        # Measure TTFB and total response time
        start_time = time.time()
        first_byte_received = False
        for chunk in chain.stream({"topic": "center of the Earth"}):
            if not first_byte_received:
                end_time_ttfb = time.time()
                ttfb = (end_time_ttfb - start_time) * 1000  # Convert to milliseconds
                total_ttfb += ttfb
                first_byte_received = True
        end_time_total = time.time()
        total_time = (end_time_total - start_time) * 1000  # Convert to milliseconds
        total_response_time += total_time

        # Add the run number, TTFB, and total response time to the table
        table.add_row([i + 1, f"{ttfb:.3f}", f"{total_time:.3f}"])

    average_ttfb = total_ttfb / num_runs
    average_response_time = total_response_time / num_runs

    print("\nBenchmark Results:")
    print("=========================================")
    # Create a table for the model information
    model_table = PrettyTable()
    model_table.field_names = ["ID", "Name", "Provider", "Mode"]
    model_table.add_row(["llama3-70b-8192", "LLaMA 3 70B", "Groq", "Streaming"])
    # Print the model information table
    print("\nTable 1: Model Information:")
    print(model_table)

    # Print the first table
    print("\nTable 2: Response Times")
    print(table)

    # Create a second table with four columns: "No of runs", "Total Time Taken", "Average TTFB", "Average TTLB"
    table2 = PrettyTable()
    table2.field_names = [
        "No of runs",
        "Total Time Taken",
        "Average TTFB",
        "Average TTLB",
    ]
    table2.add_row(
        [
            num_runs,
            f"{total_response_time / 1000:.3f} seconds",
            f"{average_ttfb:.3f} ms",
            f"{average_response_time:.3f} ms",
        ]
    )

    # Print the second table
    print("\nTable 3: Benchmark Summary")
    print(table2)


def llama3_70b_batch():
    chat = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat

    num_runs = 5
    total_response_time = 0

    # Create a table with two columns: "Run No" and "TTLB"
    table = PrettyTable()
    table.field_names = ["Run No", "TTLB (ms)"]

    for i in range(num_runs):
        # Measure total response time
        start_time = time.time()
        chain.invoke(
            {"text": "Write a short story about a journey to the center of the Earth."}
        )
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_response_time += total_time

        # Add the run number and total response time to the table
        table.add_row([i + 1, f"{total_time:.3f}"])

    average_response_time = total_response_time / num_runs

    print("\nBenchmark Results:")
    print("=========================================")
    # Create a table for the model information
    model_table = PrettyTable()
    model_table.field_names = ["ID", "Name", "Provider", "Mode"]
    model_table.add_row(["llama3-70b-8192", "LLaMA 3 70B", "Groq", "Batch"])
    # Print the model information table
    print("\nTable 1: Model Information")
    print(model_table)

    # Print the first table
    print("\nTable 2: Response Times")
    print(table)

    # Create a second table with four columns: "No of runs", "Total Time Taken", "Average TTFB", "Average TTLB"
    table2 = PrettyTable()
    table2.field_names = [
        "No of runs",
        "Total Time Taken",
        "Average TTLB",
    ]
    table2.add_row(
        [
            num_runs,
            f"{total_response_time / 1000:.3f} seconds",
            f"{average_response_time:.3f} ms",
        ]
    )

    # Print the second table
    print("\nTable 3: Benchmark Summary")
    print(table2)


def mixtral_streaming():
    chat = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a short story about a journey to the {topic}")]
    )
    chain = prompt | chat

    num_runs = 2
    total_ttfb = 0
    total_response_time = 0

    # Create a table with three columns: "Run No", "TTFB", and "Total"
    table = PrettyTable()
    table.field_names = ["Run No", "TTFB (ms)", "TTLB (ms)"]

    for i in range(num_runs):
        # Measure TTFB and total response time
        start_time = time.time()
        first_byte_received = False
        for chunk in chain.stream({"topic": "center of the Earth"}):
            if not first_byte_received:
                end_time_ttfb = time.time()
                ttfb = (end_time_ttfb - start_time) * 1000  # Convert to milliseconds
                total_ttfb += ttfb
                first_byte_received = True
        end_time_total = time.time()
        total_time = (end_time_total - start_time) * 1000  # Convert to milliseconds
        total_response_time += total_time

        # Add the run number, TTFB, and total response time to the table
        table.add_row([i + 1, f"{ttfb:.3f}", f"{total_time:.3f}"])

    average_ttfb = total_ttfb / num_runs
    average_response_time = total_response_time / num_runs

    print("\nBenchmark Results:")
    print("=========================================")
    # Create a table for the model information
    model_table = PrettyTable()
    model_table.field_names = ["ID", "Name", "Provider", "Mode"]
    model_table.add_row(["mixtral-8x7b-32768", "Mixtral 8x7b", "Groq", "Streaming"])
    # Print the model information table
    print("\nTable 1: Model Information:")
    print(model_table)

    # Print the first table
    print("\nTable 2: Response Times")
    print(table)

    # Create a second table with four columns: "No of runs", "Total Time Taken", "Average TTFB", "Average TTLB"
    table2 = PrettyTable()
    table2.field_names = [
        "No of runs",
        "Total Time Taken",
        "Average TTFB",
        "Average TTLB",
    ]
    table2.add_row(
        [
            num_runs,
            f"{total_response_time / 1000:.3f} seconds",
            f"{average_ttfb:.3f} ms",
            f"{average_response_time:.3f} ms",
        ]
    )

    # Print the second table
    print("\nTable 3: Benchmark Summary")
    print(table2)


def mixtral_batch():
    chat = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat

    num_runs = 2
    total_response_time = 0

    # Create a table with two columns: "Run No" and "TTLB"
    table = PrettyTable()
    table.field_names = ["Run No", "TTLB (ms)"]

    for i in range(num_runs):
        # Measure total response time
        start_time = time.time()
        chain.invoke(
            {"text": "Write a short story about a journey to the center of the Earth."}
        )
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_response_time += total_time

        # Add the run number and total response time to the table
        table.add_row([i + 1, f"{total_time:.3f}"])

    average_response_time = total_response_time / num_runs

    print("\nBenchmark Results:")
    print("=========================================")
    # Create a table for the model information
    model_table = PrettyTable()
    model_table.field_names = ["ID", "Name", "Provider", "Mode"]
    model_table.add_row(["mixtral-8x7b-32768", "Mixtral 8x7b", "Groq", "Batch"])
    # Print the model information table
    print("\nTable 1: Model Information")
    print(model_table)

    # Print the first table
    print("\nTable 2: Response Times")
    print(table)

    # Create a second table with four columns: "No of runs", "Total Time Taken", "Average TTFB", "Average TTLB"
    table2 = PrettyTable()
    table2.field_names = [
        "No of runs",
        "Total Time Taken",
        "Average TTLB",
    ]
    table2.add_row(
        [
            num_runs,
            f"{total_response_time / 1000:.3f} seconds",
            f"{average_response_time:.3f} ms",
        ]
    )

    # Print the second table
    print("\nTable 3: Benchmark Summary")
    print(table2)


def gemma_streaming():
    chat = ChatGroq(
        temperature=0,
        model_name="gemma-7b-it",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a very short story about a journey to the {topic}")]
    )
    chain = prompt | chat

    num_runs = 2
    total_ttfb = 0
    total_response_time = 0

    # Create a table with three columns: "Run No", "TTFB", and "Total"
    table = PrettyTable()
    table.field_names = ["Run No", "TTFB (ms)", "TTLB (ms)"]

    for i in range(num_runs):
        # Measure TTFB and total response time
        start_time = time.time()
        first_byte_received = False
        for chunk in chain.stream({"topic": "center of the Earth"}):
            if not first_byte_received:
                end_time_ttfb = time.time()
                ttfb = (end_time_ttfb - start_time) * 1000  # Convert to milliseconds
                total_ttfb += ttfb
                first_byte_received = True
        end_time_total = time.time()
        total_time = (end_time_total - start_time) * 1000  # Convert to milliseconds
        total_response_time += total_time

        # Add the run number, TTFB, and total response time to the table
        table.add_row([i + 1, f"{ttfb:.3f}", f"{total_time:.3f}"])

    average_ttfb = total_ttfb / num_runs
    average_response_time = total_response_time / num_runs

    print("\nBenchmark Results:")
    print("=========================================")
    # Create a table for the model information
    model_table = PrettyTable()
    model_table.field_names = ["ID", "Name", "Provider", "Mode"]
    model_table.add_row(["gemma-7b-it", "Gemma 7B", "Groq", "Streaming"])
    # Print the model information table
    print("\nTable 1: Model Information:")
    print(model_table)

    # Print the first table
    print("\nTable 2: Response Times")
    print(table)

    # Create a second table with four columns: "No of runs", "Total Time Taken", "Average TTFB", "Average TTLB"
    table2 = PrettyTable()
    table2.field_names = [
        "No of runs",
        "Total Time Taken",
        "Average TTFB",
        "Average TTLB",
    ]
    table2.add_row(
        [
            num_runs,
            f"{total_response_time / 1000:.3f} seconds",
            f"{average_ttfb:.3f} ms",
            f"{average_response_time:.3f} ms",
        ]
    )

    # Print the second table
    print("\nTable 3: Benchmark Summary")
    print(table2)


def gemma_batch():
    chat = ChatGroq(
        temperature=0,
        model_name="gemma-7b-it",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat

    num_runs = 2
    total_response_time = 0

    # Create a table with two columns: "Run No" and "TTLB"
    table = PrettyTable()
    table.field_names = ["Run No", "TTLB (ms)"]

    for i in range(num_runs):
        # Measure total response time
        start_time = time.time()
        chain.invoke(
            {
                "text": "Write a very short story about a journey to the center of the Earth."
            }
        )
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds
        total_response_time += total_time

        # Add the run number and total response time to the table
        table.add_row([i + 1, f"{total_time:.3f}"])

    average_response_time = total_response_time / num_runs

    print("\nBenchmark Results:")
    print("=========================================")
    # Create a table for the model information
    model_table = PrettyTable()
    model_table.field_names = ["ID", "Name", "Provider", "Mode"]
    model_table.add_row(["gemma-7b-it", "Gemma 7b", "Groq", "Batch"])
    # Print the model information table
    print("\nTable 1: Model Information")
    print(model_table)

    # Print the first table
    print("\nTable 2: Response Times")
    print(table)

    # Create a second table with four columns: "No of runs", "Total Time Taken", "Average TTFB", "Average TTLB"
    table2 = PrettyTable()
    table2.field_names = [
        "No of runs",
        "Total Time Taken",
        "Average TTLB",
    ]
    table2.add_row(
        [
            num_runs,
            f"{total_response_time / 1000:.3f} seconds",
            f"{average_response_time:.3f} ms",
        ]
    )

    # Print the second table
    print("\nTable 3: Benchmark Summary")
    print(table2)


llama3_70b_streaming()
llama3_70b_batch()
