from colour import Color
import inspect
import random
import re
import sys

import chalk
from hypothesis import given, seed as hypothesis_seed, settings
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import composite, floats, integers
import numpy as np
import tensordiagram as td
import torch

from IPython.display import display, SVG


BG_COLOR = "white"
SOLARIZED_LIGHT_COLORS = {
    "base3": "#fdf6e3",
    "base2": "#eee8d5",
    "base01": "#586e75",
    "base00": "#657b83",
    "base0": "#839496",
    "base1": "#93a1a1",
    "orange": "#cb4b16",
    "red": "#dc322f",
    "magenta": "#d33682",
    "violet": "#6c71c4",
    "blue": "#268bd2",
    "cyan": "#2aa198",
    "green": "#859900",
    "yellow": "#b58900",
}


def draw_examples(examples, diagram_fn=None):
  assert len(examples) > 0

  def opacity_fn(idx, v):
    return (abs(v) + 1.) / 11

  def to_diagram(k, t):
    if not t.shape:
      t = t[None]

    if diagram_fn and k in diagram_fn:
      return diagram_fn[k](t).to_chalk_diagram()

    if t.dtype == torch.float32:
      format_fn = lambda x: f"{x:.2f}"
    else:
      format_fn = None

    tensor_diagram = td.to_diagram(t) \
      .fill_color(SOLARIZED_LIGHT_COLORS["magenta"]) \
      .fill_opacity(opacity_fn) \
      .fill_values(format_fn=format_fn)

    return tensor_diagram.to_chalk_diagram()

  mhs = [0] * 100
  grid = {}
  for i, example in enumerate(examples):
    for k, v in example.items():
      if k not in grid:
        grid[k] = []
      diagram = to_diagram(k, v)
      if len(v.shape) == 3:
        diagram = diagram.pad(1.1)
      env = diagram.get_envelope()
      mhs[i] = max(mhs[i], env.height)
      grid[k].append(diagram)

  cols = []
  for k, vs in grid.items():
    k_diagram = chalk.text(k, 0.6).fill_color(Color("black")).line_width(0.0)
    col = [k_diagram]
    for i, d in enumerate(vs):
      env = d.get_envelope()
      col.append(d.center_xy().with_envelope(chalk.rectangle(env.width, mhs[i])))
    cols.append(chalk.vcat(col, 1.0))

  content = chalk.vstrut(0.5) / chalk.hcat(cols, 1.0).center_xy()
  final = content.center_xy().pad(1.1)
  env = final.get_envelope()
  td.set_default_height(450)
  final = chalk.rectangle(env.width, env.height).fill_color(Color(BG_COLOR)).line_width(0.0) + final
  return final


def get_shape_from_annotation(annotation):
  """
  Parses a jaxtyping annotation and returns the shape string.
  e.g., Int[T, "m n"] -> "m n"
        Int[T, ""]   -> ""
        Int[T, "*shape"] -> "*shape"
  """
  if annotation == inspect.Parameter.empty:
    return "*shape"  # default to wildcard if no annotation

  annotation_str = str(annotation)

  # regex to find the shape string inside the jaxtyping hint
  match = re.search(r'\[.+,\s*["\'](.*?)["\']\s*\]', annotation_str)

  if match:
    return match.group(1)  # return the captured string ("m n", "", "*shape")

  return "*shape"  # default to wildcard if parsing fails


def get_dtype_from_annotation(annotation):
  if annotation == inspect.Parameter.empty:
    return torch.int32

  annotation_str = str(annotation)

  if "Float" in annotation_str:
    return torch.float32
  elif "Int" in annotation_str:
    return torch.int32

  return torch.int32


@composite
def spec(draw, problem, max_dim, min_dim, max_side, min_side, max_value, min_value):
  default_shape = draw(array_shapes(min_dims=min_dim, max_dims=max_dim, min_side=min_side, max_side=max_side))

  # this dictionary will store the size for named dimensions like 'm' or 'n'
  # to ensure they are consistent across arguments (e.g., "m n" and "n").
  dim_sizes = {}

  sig = inspect.signature(problem)
  test_tensors = []

  for param_name, param in sig.parameters.items():
    shape_string = get_shape_from_annotation(param.annotation)
    dtype = get_dtype_from_annotation(param.annotation)

    if dtype == torch.float32:
      elements_strategy = floats(min_value=min_value, max_value=max_value)
      np_dtype = np.float32
    else:
      elements_strategy = integers(min_value=min_value, max_value=max_value)
      np_dtype = np.int32

    # --- handle different shape string cases ---

    if shape_string == "":
      # case 1: scalar tensor
      value = draw(elements_strategy)
      t = torch.tensor(value, dtype=dtype)

    elif shape_string == "*shape":
      # case 2: wildcard shape
      np_array = draw(arrays(shape=default_shape, dtype=np_dtype, elements=elements_strategy))
      t = torch.from_numpy(np_array).to(dtype)

    else:
      # case 3: named dimensions, annotation is e.g., "m n" or "n"
      dim_names = shape_string.split()
      shape = []
      for dim_name in dim_names:
        # get the size if we've seen this dim name before,
        # otherwise, draw a new size and store it in dim_sizes
        size = dim_sizes.setdefault(dim_name, draw(integers(min_side, max_side)))
        shape.append(size)

      # Generate the tensor with the dynamically created shape
      np_array = draw(arrays(shape=tuple(shape), dtype=np_dtype, elements=elements_strategy))
      t = torch.from_numpy(np_array).to(dtype)

    test_tensors.append(t)

  return tuple(test_tensors)


def gen_test(
    problem,
    problem_spec,
    max_dim,
    min_dim=1,
    max_side=5,
    min_side=1,
    max_value=5,
    min_value=-5,
    max_examples=100,
    constraint=lambda *x: x,
    seed=None):
  sig = inspect.signature(problem)
  return_dtype = get_dtype_from_annotation(sig.return_annotation)

  @hypothesis_seed(seed)
  @given(spec(problem, max_dim, min_dim, max_side, min_side, max_value, min_value))
  @settings(max_examples=max_examples)
  def test_problem(test_case):
    test_case = constraint(*test_case)
    if not isinstance(test_case, tuple):
        test_case = (test_case,)
    out_spec = problem_spec(*test_case)
    out = problem(*test_case)
    out_spec = out_spec.to(return_dtype)
    out = out.to(return_dtype)
    assert torch.allclose(out_spec, out), "tensors are not equal\n\ttarget:\n\t\t%s\n\tyours:\n\t\t%s" % (out_spec, out)

  return test_problem


def show(
    problem,
    problem_spec,
    max_dim=3,
    min_dim=1,
    max_side=5,
    min_side=1,
    max_value=5,
    min_value=-5,
    num_examples=3,
    constraint=lambda *x: x,
    diagram_fn=None,
    seed=None):
  examples = []
  test_cases = []

  # get parameter names from the function signature
  sig = inspect.signature(problem)
  param_names = list(sig.parameters.keys())

  @hypothesis_seed(seed)
  @given(spec(problem, max_dim, min_dim, max_side, min_side, max_value, min_value))
  @settings(max_examples=num_examples+1, database=None)
  def gen_samples(test_case):
    test_case = constraint(*test_case)
    if isinstance(test_case, tuple):
      test_cases.append(test_case)
    else:
      test_cases.append((test_case,))
  gen_samples()

  # skip first test case, which is always some base case
  for test_case in test_cases[1:num_examples+1]:
    target = problem_spec(*test_case)

    try:
      yours = problem(*test_case)
    except NotImplementedError:
      continue

    example = {}
    for param_name, value in zip(param_names, test_case):
      example[param_name] = value

    example["yours"] = yours
    example["target"] = target

    examples.append(example)

  if examples:
    diagram = draw_examples(examples, diagram_fn=diagram_fn)
    display(SVG(diagram._repr_svg_()))


def check(
    problem,
    problem_spec,
    max_dim=5,
    min_dim=1,
    max_side=5,
    min_side=1,
    max_value=5,
    min_value=-5,
    max_examples=5,
    constraint=lambda *x: x,
    seed=None):
    test = gen_test(
        problem,
        problem_spec,
        max_dim=max_dim,
        min_dim=min_dim,
        max_side=max_side,
        min_side=min_side,
        max_value=max_value,
        min_value=min_value,
        max_examples=max_examples,
        constraint=constraint)
    try:
      test()
    except AssertionError as e:
      print(f"❌ Test failed: {e}", file=sys.stderr)
      return
    # generate a random puppy video if you are correct.
    print("✅ Correct !")
    from IPython.display import HTML
    pups = [
    "2m78jPG",
    "pn1e9TO",
    "MQCIwzT",
    "udLK6FS",
    "ZNem5o3",
    "DS2IZ6K",
    "aydRUz8",
    "MVUdQYK",
    "k5jALH0",
    "wScLiVz",
    "Z0TII8i",
    "F1SChho",
    "9hRi2jN",
    "lvzRF3W",
    "fqHxOGI",
    "1xeUYme",
    "6tVqKyM",
    "CCxZ6Wr",
    "lMW0OPQ",
    "wHVpHVG",
    "Wj2PGRl",
    "HlaTE8H",
    "k5jALH0",
    "3V37Hqr",
    "Eq2uMTA",
    "Vy9JShx",
    "g9I2ZmK",
    "Nu4RH7f",
    "sWp0Dqd",
    "bRKfspn",
    "qawCMl5",
    "2F6j2B4",
    "fiJxCVA",
    "pCAIlxD",
    "zJx2skh",
    "2Gdl1u7",
    "aJJAY4c",
    "ros6RLC",
    "DKLBJh7",
    "eyxH0Wc",
    "rJEkEw4"]
    return HTML("""
    <video alt=\"test\" controls autoplay=1>
        <source src=\"https://openpuppies.com/mp4/%s.mp4\"  type=\"video/mp4\">
    </video>
    """%(random.sample(pups, 1)[0]))
