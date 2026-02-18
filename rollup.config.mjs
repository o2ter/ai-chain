import _ from 'lodash';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import commonjs from '@rollup/plugin-commonjs';
import babel from '@rollup/plugin-babel';
import json from '@rollup/plugin-json';
import dts from 'rollup-plugin-dts';

const rollupPlugins = [
  typescript({
    declaration: false,
    exclude: ['tests/**/*'],
  }),
  babel({
    babelrc: false,
    exclude: 'node_modules/**',
    babelHelpers: 'bundled',
  }),
  commonjs({
    transformMixedEsModules: true,
  }),
  json(),
];

const rollupConfig = {
  input: 'src/index',
  external: [
    /node_modules/,
  ],
  makeAbsoluteExternalsRelative: true,
};

export default [
  {
    ...rollupConfig,
    output: [
      {
        file: 'dist/index.js',
        format: 'cjs',
        sourcemap: true,
        exports: 'named',
      },
      {
        file: 'dist/index.mjs',
        format: 'es',
        sourcemap: true,
      },
    ],
    plugins: [
      resolve({
        extensions: [
          '.tsx', '.jsx',
          '.ts', '.mjs', '.js',
        ]
      }),
      ...rollupPlugins
    ],
  },
  {
    ...rollupConfig,
    output: [
      {
        file: `dist/index.d.ts`,
        format: 'es',
      },
    ],
    plugins: [
      resolve({
        extensions: [
          '.tsx', '.jsx',
          '.ts', '.mjs', '.js',
        ]
      }),
      dts()
    ],
  }
];