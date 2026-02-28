FROM python:3.11.4-slim-bullseye as prod


RUN pip install poetry==1.8.2

# Configuring poetry
RUN poetry config virtualenvs.create false
RUN poetry config cache-dir /tmp/poetry_cache

# Copying requirements of a project
COPY pyproject.toml poetry.lock /app/src/
WORKDIR /app/src

# Installing requirements
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main

# Patch pygadm: the NAME_i assignment loop is redundant (those columns are
# dropped and re-added by the merge 3 lines later) and crashes with a length
# mismatch when complete_df has more rows than level_gdf. Remove those 2 lines.
RUN python -c "\
import pathlib; \
p = pathlib.Path('/usr/local/lib/python3.11/site-packages/pygadm/__init__.py'); \
src = p.read_text(); \
bad = '\n        for i in range(int(content_level) + 1):\n            level_gdf.loc[:, f\"NAME_{i}\"] = complete_df[f\"NAME_{i}\"].values\n'; \
fixed = src.replace(bad, '\n'); \
p.write_text(fixed); \
print('pygadm patched' if fixed != src else 'WARNING: pattern not found') \
"

# Copying actuall application
COPY . /app/src/
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main

CMD ["/usr/local/bin/python", "-m", "waterpath_data_service"]

FROM prod as dev

RUN --mount=type=cache,target=/tmp/poetry_cache poetry install
