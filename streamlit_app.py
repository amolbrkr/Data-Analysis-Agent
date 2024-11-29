import json
import base64
import os

import duckdb
import pandas as pd
import plotly.express as px
import sqlalchemy
import streamlit as st
from openai import OpenAI


class DataChatAgent:
    def __init__(self):
        # Initialize session state
        if "database_connection" not in st.session_state:
            st.session_state.database_connection = None
        if "tables_info" not in st.session_state:
            st.session_state.tables_info = {}
        if "database_type" not in st.session_state:
            st.session_state.database_type = None

        self.client = OpenAI(api_key=st.secrets["openai_key"])
        print("🚀 Initialized DataChatAgent")

        # Initialize messages with enhanced structure if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Message structure will be:
        # {
        #     "role": "user" | "assistant",
        #     "content": str,
        #     "display_type": "text" | "sql" | "dataframe" | "error" | "plot",
        #     "data": Optional[Any]  # Contains dataframe, plot object, etc.
        # }

        # Updated tools schema with single function
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_and_execute_sql",
                    "description": "Generate and execute a SQL query based on a user's question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_input": {
                                "type": "string",
                                "description": "The user's question or request about the data.",
                            }
                        },
                        "required": ["user_input"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_visualization",
                    "description": "Generate a visualization based on the user's request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_input": {
                                "type": "string",
                                "description": "The user's request for visualization",
                            }
                        },
                        "required": ["user_input"],
                    },
                },
            },
        ]

        # Add memory initialization
        if "memories" not in st.session_state:
            st.session_state.memories = {}

    def connect_to_database(self, connection_type, **kwargs):
        """Establishes database connection"""
        try:
            print(f"📡 Attempting to connect to {connection_type} database")
            if connection_type == "duckdb":
                st.session_state.database_connection = duckdb.connect(
                    database=":memory:", read_only=False
                )
                st.session_state.database_type = "duckdb"
            elif connection_type == "postgres":
                connection_string = f"postgresql://{kwargs['user']}:{kwargs['password']}@{kwargs['host']}:{kwargs['port']}/{kwargs['database']}"
                print(
                    f"🔌 Connecting to PostgreSQL at {kwargs['host']}:{kwargs['port']}/{kwargs['database']}"
                )
                engine = sqlalchemy.create_engine(connection_string)
                st.session_state.database_connection = engine
                st.session_state.database_type = "postgres"
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {str(e)}")
            return False

    def load_file_to_database(self, uploaded_file):
        """Loads uploaded file into database"""
        try:
            print(f"📂 Processing uploaded file: {uploaded_file.name}")
            file_name = uploaded_file.name.split(".")[0]
            file_extension = uploaded_file.name.split(".")[-1].lower()

            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            else:
                print(f"❌ Unsupported file format: {file_extension}")
                return False, "Unsupported file format"

            print(f"📊 Loaded dataframe with shape: {df.shape}")
            st.session_state.database_connection.register(file_name, df)

            st.session_state.tables_info[file_name] = {
                "columns": list(df.columns),
                "shape": df.shape,
                "preview": df.head(),
                "dtypes": df.dtypes.to_dict(),
            }

            self._update_schema_string()
            print(f"✅ Successfully loaded {file_name} into database")
            return True, file_name

        except Exception as e:
            print(f"❌ File loading failed: {str(e)}")
            return False, str(e)

    def _get_sql_type(self, dtype):
        """Convert pandas dtype to SQL type"""
        dtype_str = str(dtype)
        if "int" in dtype_str:
            return "INTEGER"
        elif "float" in dtype_str:
            return "FLOAT"
        elif "datetime" in dtype_str:
            return "DATETIME"
        else:
            return "VARCHAR"

    def _create_schema_prompt(self):
        """Creates a schema prompt with CREATE TABLE statements and sample data"""
        print("🔍 Generating schema prompt")
        schema_statements = []

        # Use appropriate quotes based on database type
        quote = '"' if st.session_state.database_type == "duckdb" else "`"

        for table_name, info in st.session_state.tables_info.items():
            # Add sample data preview
            preview_df = info["preview"]
            sample_data = preview_df.to_string(index=False, max_rows=3)

            columns = []
            for col, dtype in info["dtypes"].items():
                sql_type = self._get_sql_type(dtype)
                columns.append(f"    {quote}{col}{quote} {sql_type}")

            create_table = f"""
            CREATE TABLE {quote}{table_name}{quote} ({',\n'.join(columns)});
            
            -- Sample data for {table_name}:
            {sample_data}"""
            schema_statements.append(create_table)

        final_schema = "\n".join(schema_statements)
        print(f"📋 Generated schema:\n{final_schema}")
        return final_schema

    def generate_and_execute_sql(self, user_input):
        """Generates SQL query from natural language input and executes it"""
        try:
            print(f"🤔 Processing user question: {user_input}")

            # Check if database is ready
            if not st.session_state.tables_info:
                return {
                    "type": "error",
                    "content": "No tables available. Please upload data first.",
                }

            # Generate SQL query
            schema_prompt = self._create_schema_prompt()
            quote_style = (
                'double quotes (")'
                if st.session_state.database_type == "duckdb"
                else "backticks (`)"
            )

            prompt = f"""Given the following SQL tables, your job is to write queries given a user's request.

            {schema_prompt}

            Important notes:
            1. Use {quote_style} around table and column names to handle special characters
            2. Always fully qualify column names with table names
            3. The query must use only the tables and columns shown above
            4. Use proper table aliases if needed

            User Question: {user_input}

            The query should work with {'PostgreSQL' if st.session_state.database_type == 'postgres' else 'DuckDB'} syntax
            Return the ouput strictly as an SQL query without any backticks, quotes, or other formatting."""

            # Get SQL query from OpenAI
            messages = [
                {
                    "role": "system",
                    "content": "You are a SQL expert. Generate only the SQL query without any explanations.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0
            )

            # Validate OpenAI response
            if not response.choices or not response.choices[0].message.content:
                return {"type": "error", "content": "Failed to generate SQL query"}

            query = response.choices[0].message.content.strip()

            print(f"✨ Generated SQL query:\n{query}")

            # Execute the query
            try:
                print(f"▶️ Executing query:\n{query}")
                result = st.session_state.database_connection.execute(query).fetch_df()
                print(f"✅ Query executed successfully. Result shape: {result.shape}")

                return {"type": "sql", "query": query, "results": result}

            except Exception as exec_error:
                error_msg = str(exec_error)
                print(f"❌ Query execution failed: {error_msg}")

                # Enhanced error messages
                if (
                    "column not found" in error_msg.lower()
                    or "not found in from clause" in error_msg.lower()
                ):
                    available_columns = []
                    for table_name, info in st.session_state.tables_info.items():
                        available_columns.extend(
                            [f"`{table_name}`.`{col}`" for col in info["columns"]]
                        )
                    error_msg = f"Column not found. Available columns are: {', '.join(available_columns)}"

                return {
                    "type": "error",
                    "query": query,
                    "content": f"Error executing query: {error_msg}",
                }

        except Exception as e:
            print(f"❌ Error in generate_and_execute_sql: {str(e)}")
            return {"type": "error", "content": str(e)}

    def get_visualization_details(self, user_input):
        """Get visualization type and required columns from LLM"""
        schema_prompt = self._create_schema_prompt()

        prompt = f"""Given the following database schema and user request, determine the appropriate visualization type and required columns.

        {schema_prompt}

        User Request: {user_input}

        Return only a JSON object with two fields:
        1. 'viz_type': One of ['histogram', 'bar', 'scatter', 'line', 'box', 'pie']
        2. 'columns': List of required column names, fully qualified with table names

        Example response:
        {{"viz_type": "scatter", "columns": ["table1.x_column", "table1.y_column"]}}"""

        messages = [
            {
                "role": "system",
                "content": "You are a data visualization expert. Respond only with the requested JSON format.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ Error parsing visualization response: {str(e)}")
            return None

    def create_visualization(self, viz_type, data):
        """Create the appropriate plotly visualization"""
        try:
            if viz_type == "histogram":
                return px.histogram(
                    data, x=data.columns[0], title=f"Distribution of {data.columns[0]}"
                )

            elif viz_type == "bar":
                return px.bar(
                    data,
                    x=data.columns[0],
                    y=data.columns[1],
                    title=f"{data.columns[1]} by {data.columns[0]}",
                )

            elif viz_type == "scatter":
                return px.scatter(
                    data,
                    x=data.columns[0],
                    y=data.columns[1],
                    title=f"{data.columns[1]} vs {data.columns[0]}",
                    trendline="ols",
                )

            elif viz_type == "line":
                return px.line(
                    data,
                    x=data.columns[0],
                    y=data.columns[1],
                    title=f"{data.columns[1]} over {data.columns[0]}",
                )

            elif viz_type == "box":
                return px.box(
                    data,
                    x=data.columns[0],
                    y=data.columns[1],
                    title=f"Distribution of {data.columns[1]} by {data.columns[0]}",
                )

            elif viz_type == "pie":
                return px.pie(
                    data,
                    values=data.columns[1],
                    names=data.columns[0],
                    title=f"{data.columns[1]} Distribution",
                )

            return None
        except Exception as e:
            print(f"❌ Error creating visualization: {str(e)}")
            return None

    def generate_visualization(self, user_input):
        """Main visualization function called by the LLM"""
        try:
            print("Generating visualization called.")

            # Get visualization details from LLM
            viz_details = self.get_visualization_details(user_input)
            if not viz_details:
                return {
                    "type": "error",
                    "content": "Failed to determine visualization type",
                }

            print("Viz details:", viz_details)

            # Generate SQL to get required data
            columns_str = ", ".join(viz_details["columns"])
            query = (
                f"SELECT {columns_str} FROM {viz_details['columns'][0].split('.')[0]}"
            )

            print("Query:", query)

            # Execute query
            result = st.session_state.database_connection.execute(query).fetch_df()

            print("Result:", result)

            # Create visualization
            plot = self.create_visualization(viz_details["viz_type"], result)

            print("Plot from create_visualization:", plot)

            if plot is None:
                return {"type": "error", "content": "Failed to create visualization"}

            # Save plot image
            image_base64 = self._save_plot_image(plot)

            return {
                "type": "plot",
                "content": f"Here's a {viz_details['viz_type']} visualization:",
                "data": plot,
                "image_base64": image_base64,
            }

        except Exception as e:
            print(f"❌ Error in generate_visualization: {str(e)}")
            return {"type": "error", "content": str(e)}

    def _update_schema_string(self):
        """
        Updates the schema string used for SQL generation
        """
        print("🔄 Updating schema string")
        try:
            schema = []
            for table_name, info in st.session_state.tables_info.items():
                print(f"📝 Processing schema for table: {table_name}")

                # Get column definitions with types
                columns = []
                for col, dtype in info["dtypes"].items():
                    sql_type = self._get_sql_type(dtype)
                    columns.append(f"{col} {sql_type}")

                # Create table schema string
                table_schema = (
                    f"""CREATE TABLE {table_name} ({',\n    '.join(columns)});"""
                )
                schema.append(table_schema)

            # Store the complete schema string in session state
            st.session_state.table_schemas = "\n\n".join(schema)
            print(f"✅ Updated schema string:\n{st.session_state.table_schemas}")

        except Exception as e:
            print(f"❌ Error updating schema string: {str(e)}")
            st.session_state.table_schemas = ""

    def get_tables_info_text(self):
        """Generate natural language description of loaded tables"""
        if not st.session_state.tables_info:
            return "There are no tables currently loaded in the database."

        info_text = []
        info_text.append(
            f"There are {len(st.session_state.tables_info)} tables currently loaded:"
        )

        for table_name, info in st.session_state.tables_info.items():
            rows, cols = info["shape"]
            info_text.append(
                f"\n- Table '{table_name}' with {rows:,} rows and {cols} columns:"
            )
            info_text.append(f"  Columns: {', '.join(info['columns'])}")
            info_text.append(f"  Sample data:\n{info['preview'].to_string()}\n")

        return "\n".join(info_text)

    def _infer_memory_action(self, user_input, context=""):
        """Determine if and how to handle memory for the given input"""
        try:
            prompt = f"""Analyze if this user input contains important information that should be stored for future reference.
            
            Context: {context}
            User Input: {user_input}

            If the input contains important findings, insights, or preferences that should be remembered, return a JSON with:
            {{"action": "store", "key": "<descriptive_key>", "value": "<concise_summary>"}}

            If we should retrieve related memories, return:
            {{"action": "retrieve", "key": "<search_term>"}}

            Otherwise return:
            {{"action": "none"}}

            Focus on storing analytical insights, key findings, and user preferences.
            Keys should be descriptive and categorized (e.g., "sales_insight", "data_preference", "key_metric")."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ Error in memory inference: {str(e)}")
            return {"action": "none"}

    def _handle_memory(self, user_input, context=""):
        """Process memory actions for user input"""
        memory_action = self._infer_memory_action(user_input, context)
        
        if memory_action["action"] == "store":
            st.session_state.memories[memory_action["key"]] = {
                "value": memory_action["value"],
                "timestamp": pd.Timestamp.now().isoformat(),
            }
            print(f"💾 Stored memory: {memory_action['key']} = {memory_action['value']}")
            
        elif memory_action["action"] == "retrieve":
            # Filter memories based on key similarity
            relevant_memories = {
                k: v for k, v in st.session_state.memories.items() 
                if memory_action["key"].lower() in k.lower()
            }
            if relevant_memories:
                memory_context = "\n".join([
                    f"- {k}: {v['value']}" for k, v in relevant_memories.items()
                ])
                return f"{user_input}\n\nRelevant past findings:\n{memory_context}"
        
        return user_input

    def process_user_input(self):
        """Process user input using OpenAI function calling"""
        try:
            print(f"🤔 Processing user input")

            if not st.session_state.tables_info:
                return {
                    "type": "text",
                    "content": "No tables available. Please upload data first.",
                }

            context = self.get_tables_info_text()
            conversation_history = [
                {
                    "role": "system",
                    "content": """You are a data analysis assistant. Your job is to help users analyze their data by: \n1. Understanding their questions about the data \n2. Using SQL queries to answer their questions \n3. Providing analysis and insights based on the data \n\nWhen appropriate use the custom functions provided to you. You do not have to call them every time, only when needed.""",
                },
                {
                    "role": "user",
                    "content": f"Current database state:\n{context}",
                },
            ]

            for message in st.session_state.messages[-5:]:
                if message["display_type"] == "plot":
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": message["content"]},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{message['image_base64']}"
                                    },
                                },
                            ],
                        }
                    )
                elif message["display_type"] == "sql":
                    conversation_history.append(
                        {
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"SQL Query: {message['content']} \n {message['data'].to_string()}",
                                },
                            ],
                        }
                    )
                elif message["display_type"] == "error":  # Ignore errors
                    pass
                else:
                    conversation_history.append(
                        {"role": message["role"], "content": message["content"]}
                    )

            print("Conversation history: ", conversation_history)

            # Add memory handling before processing
            modified_input = self._handle_memory(conversation_history[-1]["content"], context)
            
            # Update conversation history with modified input
            conversation_history.append({"role": "user", "content": modified_input})

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                tools=self.tools,
                tool_choice="auto",
                temperature=0,
            )

            message = response.choices[0].message

            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "generate_and_execute_sql":
                    return self.generate_and_execute_sql(function_args["user_input"])
                elif function_name == "generate_visualization":
                    return self.generate_visualization(function_args["user_input"])
            else:
                return {
                    "type": "text",
                    "content": (
                        message.content if message.content else "No response generated"
                    ),
                }

        except Exception as e:
            print(f"❌ Error in process_user_input: {str(e)}")
            return {"type": "error", "content": str(e)}

    def _save_plot_image(self, fig):
        """Save plotly figure as image and return base64 encoding"""

        return base64.b64encode(fig.to_image()).decode("ascii")


def main():
    st.title("Dana: The Interactive Data Assistant")

    app = DataChatAgent()

    # Database Connection Section
    st.sidebar.header("Database Connection")
    connection_type = st.sidebar.radio(
        "Select Connection Type", ["File Upload", "PostgreSQL"]
    )

    if connection_type == "File Upload":
        uploaded_files = st.sidebar.file_uploader(
            "Upload CSV or Excel files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            app.connect_to_database("duckdb")
            for file in uploaded_files:
                success, message = app.load_file_to_database(file)
                if success:
                    st.sidebar.success(f"Loaded: {message}")
                else:
                    st.sidebar.error(message)

    else:
        # PostgreSQL connection form
        with st.sidebar.form("postgres_connection"):
            host = st.text_input("Host")
            port = st.text_input("Port", "5432")
            database = st.text_input("Database")
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Connect"):
                if app.connect_to_database(
                    "postgres",
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                ):
                    st.sidebar.success("Connected to PostgreSQL!")

    # Add memories display in sidebar
    if st.session_state.get("memories"):
        st.sidebar.header("📝 Stored Memories")
        for key, memory in st.session_state.memories.items():
            with st.sidebar.expander(f"🔍 {key}"):
                st.write(f"**Value:** {memory['value']}")
                st.write(f"**Stored:** {pd.Timestamp(memory['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                if st.button("🗑️ Delete", key=f"delete_{key}"):
                    del st.session_state.memories[key]
                    st.rerun()

    # Display table information in sidebar
    if st.session_state.tables_info:
        st.sidebar.header("Available Tables")
        for table_name, info in st.session_state.tables_info.items():
            with st.sidebar.expander(f"📊 {table_name}"):
                st.write("**Preview:**")
                st.dataframe(info["preview"])
                st.write(
                    f"**Shape:** {info['shape'][0]} rows, {info['shape'][1]} columns"
                )
                st.write("**Columns:**", ", ".join(info["columns"]))

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display content based on type
            if message["display_type"] == "text":
                st.markdown(message["content"])

            elif message["display_type"] == "sql":
                # Display SQL query
                st.markdown(message["content"])
                # Display results dataframe
                if message.get("data") is not None:
                    st.dataframe(message["data"])

            elif message["display_type"] == "error":
                st.error(message["content"])

            elif message["display_type"] == "plot":
                st.markdown(message["content"])
                if message.get("data") is not None:
                    print("Plotting data.")
                    st.plotly_chart(message["data"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "display_type": "text", "data": None}
        )

        # Process the input
        response = app.process_user_input()

        # Display assistant response
        with st.chat_message("assistant"):
            if response["type"] == "sql":
                content = f"""I'll help you with that query! \n ```sql\n{response['query']}\n```\nHere are the results:"""
                st.markdown(content)
                st.dataframe(response["results"])

                # Save message with both text and dataframe
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "display_type": "sql",
                        "data": response["results"],
                    }
                )

            elif response["type"] == "text":
                st.markdown(response["content"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["content"],
                        "display_type": "text",
                        "data": None,
                    }
                )

            elif response["type"] == "plot":
                st.markdown(response["content"])
                if response.get("data") is not None:
                    st.plotly_chart(response["data"])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["content"],
                            "display_type": "plot",
                            "data": response["data"],
                            "image_base64": response["image_base64"],
                        }
                    )

            else:  # error
                st.error(response["content"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["content"],
                        "display_type": "error",
                        "data": None,
                    }
                )


if __name__ == "__main__":
    main()
