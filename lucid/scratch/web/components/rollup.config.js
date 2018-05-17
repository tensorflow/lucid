import svelte from 'rollup-plugin-svelte';
import resolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';

export default {
  output: {
    format: 'iife',
  },
  plugins: [
    svelte({
      dev: false,
      store: true,
      cascade: false
    }),
    resolve(),
    commonjs(),
  ]
};
