import json
import base64
import os
import traceback

import duckdb
import pandas as pd
import plotly.express as px
import sqlalchemy
import streamlit as st
from openai import OpenAI
import plotly.graph_objects as go


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
            {
                "type": "function",
                "function": {
                    "name": "analyze_correlations",
                    "description": "Analyze and visualize correlations between numeric columns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "The name of the table to analyze",
                            }
                        },
                        "required": ["table_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_summary_statistics",
                    "description": "Generate summary statistics for specified columns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "The name of the table to analyze"
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of column names to analyze. If empty, analyzes all columns."
                            }
                        },
                        "required": ["table_name"]
                    }
                }
            }
        ]

        # Add memory initialization
        if "memories" not in st.session_state:
            st.session_state.memories = {}

        # Add response length preference to session state
        if "response_length" not in st.session_state:
            st.session_state.response_length = "Concise"  # default to concise

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
            print(f"❌ Database connection failed: {str(e)}\n{traceback.format_exc()}")
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
            print(f"❌ File loading failed: {str(e)}\n{traceback.format_exc()}")
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
            print(
                f"❌ Error parsing visualization response: {str(e)}\n{traceback.format_exc()}"
            )
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
            print(
                f"❌ Error creating visualization: {str(e)}\n{traceback.format_exc()}"
            )
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
            print(
                f"❌ Error in generate_visualization: {str(e)}\n{traceback.format_exc()}"
            )
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
            print(
                f"❌ Error updating schema string: {str(e)}\n{traceback.format_exc()}"
            )
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
            print(f"❌ Error in memory inference: {str(e)}\n{traceback.format_exc()}")
            return {"action": "none"}

    def _handle_memory(self, user_input, context=""):
        """Process memory actions for user input"""
        memory_action = self._infer_memory_action(user_input, context)

        if memory_action["action"] == "store":
            st.session_state.memories[memory_action["key"]] = {
                "value": memory_action["value"],
                "timestamp": pd.Timestamp.now().isoformat(),
            }
            print(
                f"💾 Stored memory: {memory_action['key']} = {memory_action['value']}"
            )

        elif memory_action["action"] == "retrieve":
            # Filter memories based on key similarity
            relevant_memories = {
                k: v
                for k, v in st.session_state.memories.items()
                if memory_action["key"].lower() in k.lower()
            }
            if relevant_memories:
                memory_context = "\n".join(
                    [f"- {k}: {v['value']}" for k, v in relevant_memories.items()]
                )
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
            modified_input = self._handle_memory(
                conversation_history[-1]["content"], context
            )

            # Update conversation history with modified input
            conversation_history.append({"role": "user", "content": modified_input})

            max_tokens = 350 if st.session_state.response_length == "Concise" else 1200

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                tools=self.tools,
                tool_choice="auto",
                temperature=0,
                max_completion_tokens=max_tokens,
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
                elif function_name == "analyze_correlations":
                    return self.analyze_correlations(function_args["table_name"])
                elif function_name == "generate_summary_statistics":
                    return self.generate_summary_statistics(function_args["table_name"], function_args.get("columns", []))
            else:
                return {
                    "type": "text",
                    "content": (
                        message.content if message.content else "No response generated"
                    ),
                }

        except Exception as e:
            print(f"❌ Error in process_user_input: {str(e)}\n{traceback.format_exc()}")
            return {"type": "error", "content": str(e)}

    def _save_plot_image(self, fig):
        """Save plotly figure as image and return base64 encoding"""

        return base64.b64encode(fig.to_image()).decode("ascii")

    def analyze_correlations(self, table_name):
        """Analyze and visualize correlations between numeric columns"""
        try:
            print(f"📊 Analyzing correlations for table: {table_name}")

            # Get all numeric columns
            query = f'SELECT * FROM "{table_name}"'
            df = st.session_state.database_connection.execute(query).fetch_df()
            numeric_df = df.select_dtypes(include=["int64", "float64"])

            if numeric_df.empty:
                return {
                    "type": "error",
                    "content": f"No numeric columns found in table {table_name}",
                }

            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()

            # Create heatmap using plotly
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu",
                aspect="auto",
            )

            # Update layout for better readability
            fig.update_layout(
                title=f"Correlation Matrix for {table_name}",
                xaxis_title="Features",
                yaxis_title="Features",
                width=800,
                height=800,
            )

            # Save plot image
            image_base64 = self._save_plot_image(fig)

            # Create correlation summary text
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.5:  # Threshold for strong correlation
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        strong_correlations.append(f"{col1} vs {col2}: {corr:.2f}")

            summary = (
                "Strong correlations found:\n" + "\n".join(strong_correlations)
                if strong_correlations
                else "No strong correlations found."
            )

            return {
                "type": "plot",
                "content": f"Here's the correlation analysis for {table_name}:\n\n{summary}",
                "data": fig,
                "image_base64": image_base64,
            }

        except Exception as e:
            print(
                f"❌ Error in correlation analysis: {str(e)}\n{traceback.format_exc()}"
            )
            return {"type": "error", "content": str(e)}

    def generate_summary_statistics(self, table_name, columns=None):
        try:
            print(f"📊 Generating summary statistics for table: {table_name}")
            
            # Fetch data
            if columns and len(columns) > 0:
                columns_str = ', '.join(f'"{col}"' for col in columns)
                query = f'SELECT {columns_str} FROM "{table_name}"'
            else:
                query = f'SELECT * FROM "{table_name}"'
            
            df = st.session_state.database_connection.execute(query).fetch_df()
            
            summary_stats = {}
            visualizations = []
            
            # Process each column
            for col in df.columns:
                col_stats = {}
                
                # Basic statistics for all columns
                col_stats['count'] = len(df[col])
                col_stats['null_count'] = df[col].isnull().sum()
                col_stats['null_percentage'] = (col_stats['null_count'] / col_stats['count'] * 100)
                col_stats['unique_values'] = df[col].nunique()
                
                # Type-specific statistics
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats = df[col].describe()
                    col_stats.update({
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max'],
                        '25%': stats['25%'],
                        '50%': stats['50%'],
                        '75%': stats['75%']
                    })
                    
                    # Create distribution plot
                    fig = px.histogram(
                        df, x=col,
                        title=f'Distribution of {col}',
                        marginal='box'  # Add box plot on the margin
                    )
                    visualizations.append((col, fig))
                    
                elif pd.api.types.is_string_dtype(df[col]):
                    # For categorical columns
                    value_counts = df[col].value_counts().head(10)
                    col_stats['top_values'] = dict(value_counts)
                    
                    # Create bar plot of top categories
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f'Top 10 Categories in {col}'
                    )
                    visualizations.append((col, fig))
                    
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    # For datetime columns
                    col_stats.update({
                        'min_date': df[col].min(),
                        'max_date': df[col].max(),
                        'date_range_days': (df[col].max() - df[col].min()).days
                    })
                    
                    # Create timeline distribution
                    fig = px.histogram(
                        df, x=col,
                        title=f'Timeline Distribution of {col}'
                    )
                    visualizations.append((col, fig))
                
                summary_stats[col] = col_stats
            
            # Generate markdown summary
            markdown_summary = "# Summary Statistics\n\n"
            for col, stats in summary_stats.items():
                markdown_summary += f"## {col}\n"
                markdown_summary += f"- **Basic Info:**\n"
                markdown_summary += f"  - Count: {stats['count']:,}\n"
                markdown_summary += f"  - Null Values: {stats['null_count']:,} ({stats['null_percentage']:.2f}%)\n"
                markdown_summary += f"  - Unique Values: {stats['unique_values']:,}\n"
                
                if 'mean' in stats:
                    markdown_summary += f"- **Numerical Statistics:**\n"
                    markdown_summary += f"  - Mean: {stats['mean']:.2f}\n"
                    markdown_summary += f"  - Std Dev: {stats['std']:.2f}\n"
                    markdown_summary += f"  - Min: {stats['min']:.2f}\n"
                    markdown_summary += f"  - Max: {stats['max']:.2f}\n"
                    markdown_summary += f"  - Quartiles: {stats['25%']:.2f} (25%), {stats['50%']:.2f} (50%), {stats['75%']:.2f} (75%)\n"
                
                if 'top_values' in stats:
                    markdown_summary += f"- **Top Categories:**\n"
                    for val, count in stats['top_values'].items():
                        markdown_summary += f"  - {val}: {count:,}\n"
                
                if 'min_date' in stats:
                    markdown_summary += f"- **Date Range:**\n"
                    markdown_summary += f"  - From: {stats['min_date']}\n"
                    markdown_summary += f"  - To: {stats['max_date']}\n"
                    markdown_summary += f"  - Range: {stats['date_range_days']} days\n"
                
                markdown_summary += "\n"
            
            # Combine visualizations into a single figure using Figure Factory
            if visualizations:
                # Create subplot titles
                subplot_titles = [v[0] for v in visualizations]
                
                # Create a list of figures for the subplots
                figs = [v[1] for v in visualizations]
                
                # Calculate number of rows needed
                n_rows = len(visualizations)
                
                # Create subplot figure using make_subplots
                combined_fig = go.Figure()
                
                # Add each visualization as a subplot
                for idx, (name, fig) in enumerate(visualizations):
                    for trace in fig.data:
                        trace.update(
                            xaxis=f'x{idx+1}',
                            yaxis=f'y{idx+1}'
                        )
                        combined_fig.add_trace(trace)
                
                # Update layout for each subplot
                for i in range(n_rows):
                    if i == 0:
                        combined_fig.update_layout({
                            f'yaxis': {'domain': [1 - (i+1)/n_rows, 1 - i/n_rows]},
                            f'xaxis': {'anchor': f'y'}
                        })
                    else:
                        combined_fig.update_layout({
                            f'yaxis{i+1}': {'domain': [1 - (i+1)/n_rows, 1 - i/n_rows]},
                            f'xaxis{i+1}': {'anchor': f'y{i+1}'}
                        })
                
                # Update overall layout
                combined_fig.update_layout(
                    height=300 * n_rows,
                    width=800,
                    showlegend=False,
                    title_text="Distribution Plots",
                    margin=dict(t=50, b=50)
                )
                
                # Add subplot titles
                for i, title in enumerate(subplot_titles):
                    combined_fig.add_annotation(
                        text=title,
                        xref="paper",
                        yref="paper",
                        x=0,
                        y=1 - i/n_rows,
                        yanchor="bottom",
                        xanchor="left",
                        showarrow=False,
                        font=dict(size=14)
                    )
                
                # Save plot image
                image_base64 = self._save_plot_image(combined_fig)
                
                return {
                    "type": "plot",
                    "content": markdown_summary,
                    "data": combined_fig,
                    "image_base64": image_base64
                }
            else:
                return {
                    "type": "text",
                    "content": markdown_summary
                }
                
        except Exception as e:
            print(f"❌ Error in summary statistics: {str(e)}\n{traceback.format_exc()}")
            return {"type": "error", "content": str(e)}

def main():
    st.title("Dana: The Interactive Data Assistant")

    app = DataChatAgent()

    # Database Connection Section
    st.sidebar.header("🔌 Database Connection")
    connection_type = st.sidebar.radio(
        "Select Connection Type", ["File Upload", "PostgreSQL"], horizontal=True
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

    # Add response length selector in sidebar
    st.session_state.response_length = st.sidebar.radio(
        "Response Style",
        options=["Concise", "Detailed"],
        help="Choose between brief or detailed responses",
        horizontal=True,
    )

    # Display table information in sidebar
    if st.session_state.tables_info:
        st.sidebar.header("📋 Available Tables")
        for table_name, info in st.session_state.tables_info.items():
            with st.sidebar.expander(f"📊 {table_name}"):
                st.write("**Preview:**")
                st.dataframe(info["preview"])
                st.write(
                    f"**Shape:** {info['shape'][0]} rows, {info['shape'][1]} columns"
                )
                st.write("**Columns:**", ", ".join(info["columns"]))

    # Add memories display in sidebar
    if st.session_state.get("memories"):
        st.sidebar.header("📝 Stored Memories")
        for key, memory in st.session_state.memories.items():
            with st.sidebar.expander(f"🔍 {key}"):
                st.write(f"**Value:** {memory['value']}")
                st.write(
                    f"**Stored:** {pd.Timestamp(memory['timestamp']).strftime('%Y-%m-%d %H:%M')}"
                )
                if st.button("🗑️ Delete", key=f"delete_{key}"):
                    del st.session_state.memories[key]
                    st.rerun()

    # Add a debug button to print st.session_state.messages in the sidebar
    if st.sidebar.button("🐛 Debug Messages"):
        st.sidebar.write(st.session_state.messages)

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
