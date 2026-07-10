//
//  realtime.ts
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
import { RealtimeClientProvider, RealtimeSession, RealtimeSessionEvents } from '../../client/provider';
import { OpenAIRealtimeWS } from 'openai/realtime/ws';
import { RealtimeContentPart, RealtimeSessionOptions } from '../../client/types';

type OpenAIRealtimeConfig = {
  apiKey: string;
  baseURL: string;
};

export class OpenAIRealtimeProvider extends RealtimeClientProvider {

  config: OpenAIRealtimeConfig;

  constructor(config: OpenAIRealtimeConfig) {
    super();
    this.config = config;
  };

  session(options: RealtimeSessionOptions & RealtimeSessionEvents) {
    return new OpenAIRealtimeSession(this.config, options);
  }
}

class OpenAIRealtimeSession extends RealtimeSession {

  client: OpenAIRealtimeWS;

  constructor(
    config: OpenAIRealtimeConfig,
    { 
      model,
      systemMessage,
      output,
      onOpen,
      onClose,
      onError,
      ...options
    }: RealtimeSessionOptions & RealtimeSessionEvents,
  ) {
    super();
    this.client = new OpenAIRealtimeWS({ model }, config);
    this.client.socket.on('open', () => {
      this.client.send({
        type: 'session.update',
        session: {
          ...options,
          type: 'realtime',
          instructions: systemMessage,
          output_modalities: [output],
        },
      });
      if (onOpen) onOpen(this);
    });
    this.client.on('error', (error) => {
      if (onError) onError(this, error);
    });
    this.client.socket.on('close', () => {
      if (onClose) onClose(this);
    });
  }

  close() {
    this.client.close();
  }

  send(content: string | RealtimeContentPart[]) {
    this.client.send({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: _.isString(content) ? [{ type: 'input_text', text: content }] : content.map(c => {
          switch (c.type) {
            case 'text':
              return { type: 'input_text', text: c.text } as const;
            case 'image':
              return { type: 'input_image', image_url: c.image.url } as const;
            case 'audio':
              return { type: 'input_audio', audio: c.audio } as const;
            default:
              throw new Error(`Unsupported content type: ${(c as any).type}`);
          }
        }),
      },
    });
  }
}