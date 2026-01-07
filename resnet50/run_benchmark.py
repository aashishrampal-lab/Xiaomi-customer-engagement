import time
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import lax
from jax_resnet.pretrained import pretrained_resnet 

def load_imagenet_superbatch(batch_size, num_steps):
    """Loads real images and repeats them to fill the requested benchmark size."""
    total_requested = batch_size * num_steps
    print(f"Loading {total_requested} real images (repeating if necessary)...")
    
    # Using 'train' split as it is larger, and adding .repeat()
    ds = tfds.load('imagenette/320px', split='validation', as_supervised=True)
    
    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return image

    # .repeat() ensures we never run out of images before reaching total_requested
    ds = ds.map(preprocess).repeat().take(total_requested).batch(total_requested)
    
    # Convert to single numpy array
    real_images_np = next(iter(tfds.as_numpy(ds)))
    
    # This reshape will now ALWAYS succeed
    return real_images_np.reshape((num_steps, batch_size, 224, 224, 3))

#def load_imagenet_superbatch(batch_size, num_steps):
    """Loads and preprocesses real images into a single HBM-ready tensor."""
   # print(f"Loading {batch_size * num_steps} real images from Imagenette...")
    
    # We use 'imagenette' (a smaller version) for easy setup, 
    # but you can swap this for 'imagenet2012' if you have the credentials.
   # ds = tfds.load('imagenette/320px', split='validation', as_supervised=True)
    
   # def preprocess(image, label):
       # image = tf.image.resize(image, (224, 224))
      #  image = tf.cast(image, tf.float32) / 255.0
        # ImageNet Mean/Std
     #   image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    #    return image

    # Batch them to the total count we need for the benchmark
   # total_images = batch_size * num_steps
   # ds = ds.map(preprocess).take(total_images).batch(total_images)
    
    # Convert to a single numpy array
  #  real_images_np = next(iter(tfds.as_numpy(ds)))
    
    # Reshape to (steps, batch, 224, 224, 3)
def run_real_data_benchmark(model_size=50, batch_size=1024, num_steps=10):
    # 1. Setup Mesh & Sharding
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('batch',))
    data_sharding = NamedSharding(mesh, P(None, 'batch', None, None, None))
    replicated_sharding = NamedSharding(mesh, P())

    # 2. Load Model & Weights
    model_cls, variables = pretrained_resnet(size=model_size)
    model = model_cls()
    variables = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.array(x, dtype=jnp.bfloat16), replicated_sharding), 
        variables
    )

    # 3. Load REAL images and push to TPU
    real_data = load_imagenet_superbatch(batch_size, num_steps)
    real_input = jax.device_put(jnp.array(real_data, dtype=jnp.bfloat16), data_sharding)

    @jax.jit
    def pure_compute_loop(vars, inputs):
        def body_fun(carry, x_batch):
            logits = model.apply(vars, x_batch, train=False)
            return None, logits
        _, all_logits = lax.scan(body_fun, None, inputs)
        return all_logits

    # 4. Warmup & Benchmark
    print("Compiling and Warming up...")
    _ = pure_compute_loop(variables, real_input).block_until_ready()

    print(f"Running benchmark on real ImageNet data...")
    start = time.perf_counter()
    outcome = pure_compute_loop(variables, real_input).block_until_ready()
    end = time.perf_counter()

    # 5. Result
    total_time = end - start
    print(f"\n{'='*40}")
    print(f"REAL DATA TPU REPORT")
    print(f"Throughput:  {(batch_size * num_steps) / total_time:.2f} img/sec")
    print(f"Avg Latency: {(total_time / num_steps) * 1000:.2f} ms")
    print(f"{'='*40}")

if __name__ == "__main__":
    tf.config.set_visible_devices([], 'TPU')
    # Using 1024 to stay safe with memory since real images are loaded in full
    run_real_data_benchmark(batch_size=1024, num_steps=50)
