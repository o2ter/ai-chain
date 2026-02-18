//
//  index.ts
//
//  The MIT License
//  Copyright (c) 2021 - 2026 O2ter Limited. All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

import _ from 'lodash';
import { OpenAIProvider } from '../providers/openai';
import { OllamaProvider } from '../providers/ollama';
import { EmbedOptions } from './types';

const providers = {
  'openai': OpenAIProvider,
  'ollama': OllamaProvider,
} as const;

type Providers = typeof providers;
type ProviderInstances = { [x in keyof Providers]: InstanceType<Providers[x]>; };

export class Client<P extends keyof Providers> {

  #provider: ProviderInstances[P];

  constructor(format: P, options: ConstructorParameters<Providers[P]>[0]) {
    this.#provider = new providers[format](options as any) as ProviderInstances[P];
  }

  models() {
    return this.#provider.models();
  }

  embeddings(options: EmbedOptions) {
    return this.#provider.embeddings(options);
  }

  chat<S extends boolean = false>(options: Omit<Parameters<ProviderInstances[P]['chat']>[0], 'stream'> & { stream?: S }) {
    return this.#provider.chat<S>(options as any);
  }
};
